#!/usr/bin/env python3
"""
M+ Model - HotpotQA Multi-hop QA Evaluation Script
===================================================
논문 코드(longbench_pred.py) 기반으로 M+ 모델 지원 추가
L4 GPU (24GB) 메모리에 맞게 최적화

사용법:
    python eval_mplus_hotpotqa.py --num_samples 100 --split_model
    python eval_mplus_hotpotqa.py --num_samples 500 --split_model
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import re
import string
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import gc

# 현재 디렉토리를 path에 추가 (modeling_mplus import 위해)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="M+ HotpotQA Multi-hop Evaluation")
    parser.add_argument('--model_path', type=str, default='YuWangX/mplus-8b',
                        help='Model path or HuggingFace model name')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--max_gen', type=int, default=32,
                        help='Max generation length')
    parser.add_argument('--split_model', action='store_true',
                        help='Use device_map=auto for memory optimization')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results_hotpotqa',
                        help='Output directory')
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='Context chunk size for memory injection')
    parser.add_argument('--dataset', type=str, default='hotpotqa',
                        choices=['hotpotqa', '2wikimqa', 'musique'],
                        help='Multi-hop QA dataset to use')
    parser.add_argument('--reset_ltm', action='store_true',
                        help='Reset LTM between samples (recommended for independent QA evaluation)')
    return parser.parse_args()


def seed_everything(seed):
    """재현성을 위한 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def reset_ltm(model):
    """LTM을 초기 상태로 리셋 (각 샘플 독립 평가용)"""
    config = model.config
    L = model.L
    d = model.d

    # LTM 텐서들 리셋
    for idx in range(L):
        if isinstance(model.ltm, nn.ParameterList):
            # nn.ParameterList인 경우
            model.ltm[idx].data = torch.randn(
                [config.ltm_initial_size, d],
                dtype=model.memory.dtype,
                device=model.memory.device
            )
            model.ltm_keys[idx].data = torch.randn(
                [config.ltm_initial_size, config.selector_hidden_dim],
                dtype=model.memory.dtype,
                device=model.memory.device
            )
            model.ltm_recall_frequencies[idx].data = torch.zeros(
                [config.ltm_initial_size],
                dtype=model.memory.dtype,
                device=model.memory.device
            )
            model.ltm_ages[idx].data = torch.zeros(
                [config.ltm_initial_size],
                dtype=model.memory.dtype,
                device=model.memory.device
            )
        else:
            # list인 경우 (put_ltm_to_numpy 호출 후)
            model.ltm[idx] = torch.randn(
                [config.ltm_initial_size, d],
                dtype=model.memory.dtype
            ).cpu()
            model.ltm_keys[idx] = torch.randn(
                [config.ltm_initial_size, config.selector_hidden_dim],
                dtype=model.memory.dtype
            ).cpu()
            model.ltm_recall_frequencies[idx] = np.zeros([config.ltm_initial_size])
            model.ltm_ages[idx] = np.zeros([config.ltm_initial_size])

    # memory_ages 리셋
    for idx in range(L):
        model.memory_ages[idx] = np.zeros([model.num_blocks * model.num_tokens])


def normalize_answer(s):
    """정답 정규화 (SQuAD 스타일)"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """F1 스코어 계산"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Exact Match 스코어 계산"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_metrics(predictions, ground_truths):
    """전체 메트릭 계산"""
    em_scores = []
    f1_scores = []

    for pred, truths in zip(predictions, ground_truths):
        # 여러 정답 중 최고 점수 사용
        if isinstance(truths, list):
            em = max(exact_match_score(pred, truth) for truth in truths)
            f1 = max(f1_score(pred, truth) for truth in truths)
        else:
            em = exact_match_score(pred, truths)
            f1 = f1_score(pred, truths)

        em_scores.append(em)
        f1_scores.append(f1)

    return {
        'exact_match': np.mean(em_scores) * 100,
        'f1': np.mean(f1_scores) * 100,
        'total_samples': len(predictions)
    }


def load_mplus_model(model_path, split_model=False):
    """M+ 모델 로딩 (메모리 최적화)"""
    from modeling_mplus import MPlus

    print(f"Loading M+ model from {model_path}...")
    print(f"Split model: {split_model}")

    # GPU 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()

    if split_model:
        # device_map='auto'로 GPU/CPU 분산
        model = MPlus.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
    else:
        # 단일 GPU
        model = MPlus.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).cuda()

    model.eval()

    # 메모리 사용량 출력
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return model


def get_prompt_format(dataset_name):
    """데이터셋별 프롬프트 포맷"""
    prompts = {
        "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    }
    return prompts.get(dataset_name, prompts["hotpotqa"])


def run_evaluation(model, tokenizer, dataset, args):
    """메인 평가 루프"""

    prompt_format = get_prompt_format(args.dataset)

    # 모델의 device 확인
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device

    # 초기 메모리 백업 (논문 방식) - STM만 백업 (메모리 절약)
    # dtype 유지 중요 (bfloat16)
    backup_memory = model.memory.data.clone().detach()
    backup_memory_dtype = backup_memory.dtype
    backup_memory = backup_memory.cpu()

    predictions = []
    ground_truths = []
    results_detail = []

    print(f"\nEvaluating {len(dataset)} samples...")

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # STM 리셋 (논문 방식)
            model.memory.data = backup_memory.clone().detach().to(device=model.memory.device, dtype=backup_memory_dtype)

            # LTM 리셋 (옵션, OOM 방지 및 독립 평가용)
            if args.reset_ltm:
                reset_ltm(model)

            # 캐시된 드롭 메모리 정리 (OOM 방지)
            if model.cached_dropped_memories is not None:
                del model.cached_dropped_memories
                del model.cached_dropped_memory_ages
                del model.cached_dropped_keys
            model.cached_dropped_memories = None
            model.cached_dropped_memory_ages = None
            model.cached_dropped_keys = None
            model.update_step = 0

            # 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()

            # 프롬프트 생성
            prompt = prompt_format.format(context=sample['context'], input=sample['input'])

            # 토큰화
            prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids

            # 컨텍스트를 청크로 분할하여 메모리에 주입
            contexts_ids = []
            remaining_ids = prompt_ids.copy()

            # 마지막 청크는 생성을 위해 남겨둠
            while len(remaining_ids) > args.chunk_size:
                contexts_ids.append(remaining_ids[:args.chunk_size])
                remaining_ids = remaining_ids[args.chunk_size:]

            # 마지막 부분은 sentence로 사용 (생성 입력)
            sentence_ids = remaining_ids if remaining_ids else contexts_ids.pop()

            # 메모리 주입
            with torch.no_grad():
                for context_chunk in contexts_ids:
                    context_tensor = torch.tensor(context_chunk).unsqueeze(0).cuda()
                    attention_mask = torch.ones(len(context_chunk) + model.num_tokens).long().unsqueeze(0).cuda()

                    model.inject_memory(
                        context_tensor,
                        attention_mask,
                        update_memory=True
                    )

                # 생성
                sentence_tensor = torch.tensor(sentence_ids).unsqueeze(0).cuda()
                gen_attention_mask = torch.ones(
                    len(sentence_ids) + model.num_blocks * model.num_tokens
                ).unsqueeze(0).long().cuda()

                output = model.generate(
                    input_ids=sentence_tensor,
                    attention_mask=gen_attention_mask,
                    max_new_tokens=args.max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id
                )[0]

                # 생성된 부분만 추출
                pred = tokenizer.decode(output[len(sentence_ids):], skip_special_tokens=True)

            # 정답 추출
            answers = sample['answers']
            if isinstance(answers, str):
                answers = [answers]

            predictions.append(pred.strip())
            ground_truths.append(answers)

            # 상세 결과 저장
            results_detail.append({
                'idx': idx,
                'question': sample['input'],
                'prediction': pred.strip(),
                'ground_truth': answers,
                'context_length': len(prompt_ids)
            })



        except Exception as e:
            import traceback
            print(f"\nError at sample {idx}:")
            traceback.print_exc()
            predictions.append("")
            ground_truths.append(sample.get('answers', [""]))
            continue
        
    return predictions, ground_truths, results_detail


def main():
    args = parse_args()

    print("=" * 60)
    print("M+ Model - HotpotQA Multi-hop QA Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Split model: {args.split_model}")
    print("=" * 60)

    # 시드 설정
    seed_everything(args.seed)

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 데이터셋 로드
    print(f"\nLoading {args.dataset} dataset from LongBench...")

    # HuggingFace에서 직접 로드
    full_dataset = load_dataset(
        'THUDM/LongBench',
        args.dataset,
        split='test'
    )

    # 샘플 수 제한
    if args.num_samples and args.num_samples < len(full_dataset):
        indices = list(range(args.num_samples))
        dataset = full_dataset.select(indices)
    else:
        dataset = full_dataset

    print(f"Loaded {len(dataset)} samples")

    # 모델 로드
    model = load_mplus_model(args.model_path, args.split_model)
    # 논문 README 방식: 모델 경로에서 직접 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b")


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 평가 실행
    predictions, ground_truths, results_detail = run_evaluation(
        model, tokenizer, dataset, args
    )

    # 메트릭 계산
    metrics = compute_metrics(predictions, ground_truths)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Exact Match: {metrics['exact_match']:.2f}%")
    print(f"F1 Score: {metrics['f1']:.2f}%")
    print("=" * 60)

    # 결과 저장
    output_file = os.path.join(
        args.output_dir,
        f"mplus_{args.dataset}_n{args.num_samples}_seed{args.seed}.json"
    )

    results = {
        'args': vars(args),
        'metrics': metrics,
        'predictions': results_detail[:20]  # 처음 20개만 저장 (용량 절약)
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # 샘플 출력
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (first 5)")
    print("=" * 60)
    for i, detail in enumerate(results_detail[:5]):
        print(f"\n[{i+1}] Question: {detail['question'][:100]}...")
        print(f"    Prediction: {detail['prediction']}")
        print(f"    Ground Truth: {detail['ground_truth']}")


if __name__ == "__main__":
    main()
