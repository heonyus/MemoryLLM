#!/usr/bin/env python3
"""
Phase 1: M+ Retriever Co-retrieval 실험 (Full M+ Model)
========================================================
M+ 전체 모델에서 hidden states를 생성하여 retriever 평가

이전 실험과의 차이:
- 이전: Base Llama hidden states + M+ retriever weights
- 현재: M+ 전체 모델 hidden states + M+ retriever (내장)

L4 24GB에서 실행 가능 (4-bit quantization)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MPlusRetrieverFull:
    """
    M+ 전체 모델을 사용한 Retriever

    M+ 모델 내장 retriever 사용:
        - M+ 모델의 hidden states 직접 사용
        - query_proj, key_proj가 모델에 내장됨
    """

    def __init__(self, model_id="YuWangX/mplus-8b", layer_idx=16):
        print(f"\n{'='*60}")
        print(f"M+ 전체 모델 로드 (Full Model)")
        print(f"{'='*60}")

        self.layer_idx = layer_idx
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # M+ 전체 모델 로드 (4-bit, 메모리 최적화)
        print(f"\n[1/2] M+ 전체 모델 로드 중 (4-bit, low_cpu_mem)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import gc

        # GPU 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # bfloat16 → float16 (메모리 절약)
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # 디스크 오프로딩 폴더 생성
        offload_folder = "/tmp/mplus_offload"
        os.makedirs(offload_folder, exist_ok=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "20GiB", "cpu": "10GiB"},  # RAM 제한 (15GB 중 10GB만 사용)
            offload_folder=offload_folder,  # 초과분 디스크로 오프로드
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"      M+ 모델 로드 완료")

        # Retriever 컴포넌트 확인
        print(f"\n[2/2] Retriever 컴포넌트 확인...")
        layer = self.model.model.layers[layer_idx]

        self.has_builtin_retriever = hasattr(layer.self_attn, 'query_proj')

        if self.has_builtin_retriever:
            print(f"      ✓ M+ 내장 retriever 발견 (layer {layer_idx})")
            self.query_proj = layer.self_attn.query_proj
            self.key_proj = layer.self_attn.key_proj
            self.input_layernorm = layer.input_layernorm
        else:
            print(f"      ✗ 내장 retriever 없음 - safetensors에서 로드")
            self._load_retriever_from_safetensors(model_id, layer_idx)

        # GPU 메모리 확인
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"      GPU Memory: {mem:.2f} GB")
        print(f"      Layer: {layer_idx}")
        print(f"{'='*60}\n")

    def _load_retriever_from_safetensors(self, model_id, layer_idx):
        """Fallback: safetensors에서 retriever weights 로드"""
        from safetensors import safe_open
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.safetensors", "*.json"],
        )

        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)

        hidden_size = config["hidden_size"]

        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        needed_keys = [
            f"model.layers.{layer_idx}.self_attn.query_proj.0.weight",
            f"model.layers.{layer_idx}.self_attn.query_proj.2.weight",
            f"model.layers.{layer_idx}.self_attn.key_proj.0.weight",
            f"model.layers.{layer_idx}.self_attn.key_proj.2.weight",
            f"model.layers.{layer_idx}.input_layernorm.weight",
        ]

        files_to_load = {}
        for key in needed_keys:
            if key in weight_map:
                filename = weight_map[key]
                if filename not in files_to_load:
                    files_to_load[filename] = []
                files_to_load[filename].append(key)

        weights = {}
        for filename, keys in files_to_load.items():
            filepath = os.path.join(model_dir, filename)
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in keys:
                    weights[key] = f.get_tensor(key)

        # RMSNorm
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.input_layernorm.weight.data = weights[f"model.layers.{layer_idx}.input_layernorm.weight"].float()
        self.input_layernorm = self.input_layernorm.to(self.device)

        # query_proj
        q0 = weights[f"model.layers.{layer_idx}.self_attn.query_proj.0.weight"]
        q2 = weights[f"model.layers.{layer_idx}.self_attn.query_proj.2.weight"]
        self.query_proj = nn.Sequential(
            nn.Linear(q0.shape[1], q0.shape[0], bias=False),
            nn.SiLU(),
            nn.Linear(q2.shape[1], q2.shape[0], bias=False)
        )
        self.query_proj[0].weight.data = q0.float()
        self.query_proj[2].weight.data = q2.float()
        self.query_proj = self.query_proj.to(self.device)

        # key_proj
        k0 = weights[f"model.layers.{layer_idx}.self_attn.key_proj.0.weight"]
        k2 = weights[f"model.layers.{layer_idx}.self_attn.key_proj.2.weight"]
        self.key_proj = nn.Sequential(
            nn.Linear(k0.shape[1], k0.shape[0], bias=False),
            nn.SiLU(),
            nn.Linear(k2.shape[1], k2.shape[0], bias=False)
        )
        self.key_proj[0].weight.data = k0.float()
        self.key_proj[2].weight.data = k2.float()
        self.key_proj = self.key_proj.to(self.device)

    @torch.no_grad()
    def encode_tokens(self, text, proj_type="key", max_length=256):
        """
        텍스트 → token-level projected vectors

        M+ 전체 모델의 hidden states 사용
        """
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=max_length, padding=True
        ).to(self.device)

        # M+ 모델 forward → hidden states
        outputs = self.model.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )

        # layer_idx의 hidden states
        hidden = outputs.hidden_states[self.layer_idx + 1]

        # input_layernorm → projector
        normed = self.input_layernorm(hidden.float())

        if proj_type == "key":
            vecs = self.key_proj(normed)
        else:
            vecs = self.query_proj(normed)

        return vecs.squeeze(0).cpu().float(), inputs.attention_mask.squeeze(0).cpu()

    def encode_documents(self, texts, max_length=256):
        """여러 documents를 token-level key vectors로 인코딩"""
        doc_keys = []
        for text in tqdm(texts, desc="Encoding docs", leave=False):
            keys, mask = self.encode_tokens(text, "key", max_length)
            doc_keys.append((keys, mask))
        return doc_keys

    def retrieve_token_level(self, query_text, doc_keys_list, top_k=5):
        """Token-level 검색"""
        query_vecs, query_mask = self.encode_tokens(query_text, "query")
        valid_query = query_vecs[query_mask.bool()]

        doc_scores = []
        for doc_keys, doc_mask in doc_keys_list:
            valid_keys = doc_keys[doc_mask.bool()]

            if len(valid_keys) == 0:
                doc_scores.append(0.0)
                continue

            similarities = torch.sigmoid(valid_query @ valid_keys.T)
            score = similarities.mean().item()
            doc_scores.append(score)

        scores = torch.tensor(doc_scores)
        top_k_idx = torch.topk(scores, k=min(top_k, len(scores))).indices
        return top_k_idx.numpy(), scores[top_k_idx].numpy()


class LlamaRMSNorm(nn.Module):
    """Llama RMSNorm"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


def load_hotpotqa_data(num_samples=100, num_distractors=500):
    """HotpotQA 로드"""
    print(f"\n[Data] HotpotQA 로드 중...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    samples = []
    all_distractors = []
    seen_distractors = set()

    for item in tqdm(dataset, desc="Processing", leave=False):
        if len(samples) >= num_samples and len(all_distractors) >= num_distractors:
            break

        supporting_titles = set(item['supporting_facts']['title'])
        title_to_text = {}
        for title, sentences in zip(item['context']['title'], item['context']['sentences']):
            title_to_text[title] = ' '.join(sentences)

        sup_titles = [t for t in supporting_titles if t in title_to_text]
        if len(sup_titles) >= 2 and len(samples) < num_samples:
            samples.append({
                'question': item['question'],
                'answer': item['answer'],
                'doc1': title_to_text[sup_titles[0]],
                'doc2': title_to_text[sup_titles[1]],
                'title1': sup_titles[0],
                'title2': sup_titles[1]
            })

        for title, text in title_to_text.items():
            if title not in supporting_titles and title not in seen_distractors:
                if len(text) > 100:
                    all_distractors.append(text)
                    seen_distractors.add(title)

    all_distractors = all_distractors[:num_distractors]
    print(f"[Data] {len(samples)} 2-hop samples, {len(all_distractors)} distractors")
    return samples, all_distractors


def run_coretrieval_experiment(retriever, samples, distractors, gap_sizes, top_k=5, verbose_first=True):
    """Co-retrieval 실험"""
    print(f"\n{'='*70}")
    print(f"  Co-retrieval 실험 시작 (M+ Full Model)")
    print(f"{'='*70}")
    print(f"  Samples: {len(samples)}, Distractors: {len(distractors)}, Top-k: {top_k}")
    print(f"  Gap sizes: {gap_sizes}")

    results = {}
    detailed_logs = []
    first_sample_logged = False

    for gap in gap_sizes:
        print(f"\n  [Gap {gap}] 실험 중...")

        co_count = 0
        single_count = 0
        none_count = 0

        for sample_idx, sample in enumerate(tqdm(samples, desc=f"Gap={gap}", leave=False)):
            memory = []

            # Prefix
            n_prefix = min(50, len(distractors))
            prefix_idx = np.random.choice(len(distractors), n_prefix, replace=False)
            memory.extend([distractors[i] for i in prefix_idx])

            # Doc1
            doc1_idx = len(memory)
            memory.append(sample['doc1'])

            # Gap
            if gap > 0:
                n_gap = min(gap, len(distractors))
                gap_idx = np.random.choice(len(distractors), n_gap, replace=False)
                memory.extend([distractors[i] for i in gap_idx])

            # Doc2
            doc2_idx = len(memory)
            memory.append(sample['doc2'])

            # Suffix
            n_suffix = min(20, len(distractors))
            suffix_idx = np.random.choice(len(distractors), n_suffix, replace=False)
            memory.extend([distractors[i] for i in suffix_idx])

            # Encode & Retrieve
            doc_keys = retriever.encode_documents(memory)
            top_k_idx, top_k_scores = retriever.retrieve_token_level(sample['question'], doc_keys, top_k)
            top_k_set = set(top_k_idx)

            d1_found = doc1_idx in top_k_set
            d2_found = doc2_idx in top_k_set

            if d1_found and d2_found:
                co_count += 1
                result_type = "co-retrieval"
            elif d1_found or d2_found:
                single_count += 1
                result_type = "single"
            else:
                none_count += 1
                result_type = "none"

            # 상세 로깅
            if verbose_first and not first_sample_logged and sample_idx == 0:
                first_sample_logged = True
                print(f"\n  {'─'*66}")
                print(f"  [상세 로그] 첫 번째 샘플 (Gap={gap})")
                print(f"  {'─'*66}")
                print(f"\n  ▶ INPUT")
                print(f"    Question: {sample['question']}")
                print(f"    Answer: {sample['answer']}")
                print(f"\n  ▶ MEMORY: {len(memory)} docs (Doc1@{doc1_idx}, Doc2@{doc2_idx})")
                print(f"\n  ▶ OUTPUT")
                print(f"    Top-{top_k} indices: {list(top_k_idx)}")
                print(f"    Top-{top_k} scores: {[f'{s:.4f}' for s in top_k_scores]}")
                print(f"    Doc1 found: {'✓' if d1_found else '✗'}, Doc2 found: {'✓' if d2_found else '✗'}")
                print(f"    Result: {result_type.upper()}")
                print(f"  {'─'*66}")

            if sample_idx < 3:
                detailed_logs.append({
                    'gap': int(gap),
                    'sample_idx': int(sample_idx),
                    'question': sample['question'],
                    'doc1_idx': int(doc1_idx),
                    'doc2_idx': int(doc2_idx),
                    'top_k_idx': [int(i) for i in top_k_idx],
                    'top_k_scores': [float(s) for s in top_k_scores],
                    'd1_found': bool(d1_found),
                    'd2_found': bool(d2_found),
                    'result': result_type
                })

        total = len(samples)
        results[gap] = {
            'co_retrieval_rate': co_count / total,
            'single_rate': single_count / total,
            'none_rate': none_count / total,
            'counts': {'co': co_count, 'single': single_count, 'none': none_count, 'total': total}
        }

        print(f"      Co-retrieval: {co_count}/{total} ({100*co_count/total:.1f}%)")
        print(f"      Single:       {single_count}/{total} ({100*single_count/total:.1f}%)")
        print(f"      None:         {none_count}/{total} ({100*none_count/total:.1f}%)")

    return results, detailed_logs


def visualize_results(results, output_dir, config):
    """결과 시각화"""
    os.makedirs(output_dir, exist_ok=True)

    gaps = sorted(results.keys())
    co_rates = [results[g]['co_retrieval_rate'] * 100 for g in gaps]
    single_rates = [results[g]['single_rate'] * 100 for g in gaps]
    none_rates = [results[g]['none_rate'] * 100 for g in gaps]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(gaps, co_rates, 'g-o', linewidth=2, markersize=8, label='Co-retrieval')
    ax1.plot(gaps, single_rates, 'orange', linestyle='--', marker='s', linewidth=2, markersize=8, label='Single')
    ax1.plot(gaps, none_rates, 'r:', marker='^', linewidth=2, markersize=8, label='None')
    ax1.set_xlabel('Gap Size', fontsize=12)
    ax1.set_ylabel('Rate (%)', fontsize=12)
    ax1.set_title('M+ Full Model: Co-retrieval vs Gap Size', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    for x, y in zip(gaps, co_rates):
        ax1.annotate(f'{y:.0f}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    ax2 = axes[1]
    x_pos = np.arange(len(gaps))
    ax2.bar(x_pos, co_rates, 0.6, label='Co-retrieval', color='green', alpha=0.8)
    ax2.bar(x_pos, single_rates, 0.6, bottom=co_rates, label='Single', color='orange', alpha=0.8)
    ax2.bar(x_pos, none_rates, 0.6, bottom=[c+s for c, s in zip(co_rates, single_rates)], label='None', color='red', alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(g) for g in gaps])
    ax2.set_xlabel('Gap Size', fontsize=12)
    ax2.set_ylabel('Rate (%)', fontsize=12)
    ax2.set_title('Retrieval Outcome Distribution', fontsize=14)
    ax2.legend()
    ax2.set_ylim([0, 100])

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'phase1_mplus_full_coretrieval.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    from datetime import datetime
    json_path = os.path.join(output_dir, 'phase1_mplus_full_coretrieval.json')
    output_data = {
        'experiment': 'Phase 1: M+ Full Model Co-retrieval Analysis',
        'timestamp': datetime.now().isoformat(),
        'method': {
            'description': 'Full M+ model hidden states + built-in retriever',
            'model': 'YuWangX/mplus-8b (4-bit)',
            'difference': 'Uses M+ model hidden states instead of base Llama'
        },
        'config': config,
        'results': {str(k): v for k, v in results.items()}
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n  [저장] {fig_path}")
    print(f"  [저장] {json_path}")


def print_summary(results, output_dir):
    """결과 요약"""
    print(f"\n{'='*70}")
    print(f"  Phase 1: M+ Full Model Co-retrieval Analysis")
    print(f"{'='*70}")

    print(f"\n  {'Gap':>6} | {'Co-retrieval':>14} | {'Single':>12} | {'None':>10}")
    print(f"  {'-'*50}")

    for gap in sorted(results.keys()):
        r = results[gap]
        print(f"  {gap:>6} | {r['co_retrieval_rate']*100:>13.1f}% | {r['single_rate']*100:>11.1f}% | {r['none_rate']*100:>9.1f}%")

    print(f"  {'-'*50}")

    gaps = sorted(results.keys())
    if len(gaps) >= 2:
        first, last = gaps[0], gaps[-1]
        drop = results[first]['co_retrieval_rate'] - results[last]['co_retrieval_rate']
        print(f"\n  Gap {first}→{last}: Co-retrieval {results[first]['co_retrieval_rate']*100:.1f}% → {results[last]['co_retrieval_rate']*100:.1f}%")
        if results[first]['co_retrieval_rate'] > 0:
            print(f"  하락폭: {drop*100:.1f}%p ({drop/results[first]['co_retrieval_rate']*100:.0f}% 상대 감소)")

    print(f"\n{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M+ Full Model Co-retrieval Experiment")
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_distractors', type=int, default=1000)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--layer_idx', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='./experiments/results_phase1_mplus_full')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.quick:
        args.num_samples = 30
        args.num_distractors = 500
        # gap_sizes = [0, 50, 200]
        gap_sizes = [0, 10, 50]
    else:
        gap_sizes = [0, 10, 50, 100, 200, 500]

    print(f"\n{'='*60}")
    print("Phase 1: M+ Full Model Co-retrieval Experiment")
    print(f"{'='*60}")
    print(f"Samples: {args.num_samples}, Distractors: {args.num_distractors}")
    print(f"Top-k: {args.top_k}, Layer: {args.layer_idx}")
    print(f"Gaps: {gap_sizes}")

    samples, distractors = load_hotpotqa_data(args.num_samples, args.num_distractors)
    retriever = MPlusRetrieverFull(layer_idx=args.layer_idx)

    results, detailed_logs = run_coretrieval_experiment(
        retriever, samples, distractors, gap_sizes=gap_sizes, top_k=args.top_k
    )

    config = {
        'num_samples': int(len(samples)),
        'num_distractors': int(len(distractors)),
        'top_k': int(args.top_k),
        'layer_idx': int(args.layer_idx),
        'gap_sizes': [int(g) for g in gap_sizes],
        'seed': int(args.seed),
        'quick_mode': bool(args.quick),
        'detailed_logs': detailed_logs
    }

    print_summary(results, args.output_dir)
    visualize_results(results, args.output_dir, config)

    print(f"\n  실험 완료! 결과: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
