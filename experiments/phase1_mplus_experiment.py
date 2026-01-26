#!/usr/bin/env python3
"""
Phase 1: M+ Retriever Co-retrieval 실험
========================================
논문과 동일한 방식으로 M+ retriever 구현:
- input_layernorm → query_proj/key_proj
- similarity = sigmoid(dot product)

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
from safetensors import safe_open
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MPlusRetrieverProper:
    """
    논문과 동일한 M+ Retriever 구현

    논문 방식 (modeling_mplus.py:2556, 2581):
        keys = key_proj(input_layernorm(hidden_states))
        queries = query_proj(input_layernorm(hidden_states))
        scores = sigmoid(queries @ keys.T)
    """

    def __init__(self, model_id="YuWangX/mplus-8b", base_model_id="meta-llama/Llama-3.1-8B", layer_idx=16):
        print(f"\n{'='*60}")
        print(f"M+ Retriever 로드 (논문 방식)")
        print(f"{'='*60}")

        self.layer_idx = layer_idx
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Step 1: Base Llama 모델 로드 (4-bit)
        print(f"\n[1/3] Llama-3.1-8B 로드 중 (4-bit)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.base_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"      Llama 로드 완료")

        # Step 2: M+ retriever weights 로드 (safetensors에서 직접)
        print(f"\n[2/3] M+ Retriever weights 로드 중...")
        self._load_retriever_weights(model_id, layer_idx)
        print(f"      Retriever weights 로드 완료")

        # Step 3: GPU 메모리 확인
        print(f"\n[3/3] 설정 완료")
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"      GPU Memory: {mem:.2f} GB")
        print(f"      Layer: {layer_idx}")
        print(f"{'='*60}\n")

    def _load_retriever_weights(self, model_id, layer_idx):
        """safetensors에서 retriever weights 직접 로드"""

        # 모델 다운로드
        model_dir = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.safetensors", "*.json"],
        )

        # Config 로드
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)

        hidden_size = config["hidden_size"]  # 4096
        selector_dim = config["selector_hidden_dim"]  # 256

        # Index 로드
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # 필요한 weights 목록 (M+ projector는 bias 없음)
        needed_keys = [
            f"model.layers.{layer_idx}.self_attn.query_proj.0.weight",
            f"model.layers.{layer_idx}.self_attn.query_proj.2.weight",
            f"model.layers.{layer_idx}.self_attn.key_proj.0.weight",
            f"model.layers.{layer_idx}.self_attn.key_proj.2.weight",
            f"model.layers.{layer_idx}.input_layernorm.weight",
        ]

        # 파일별로 그룹화
        files_to_load = {}
        for key in needed_keys:
            if key in weight_map:
                filename = weight_map[key]
                if filename not in files_to_load:
                    files_to_load[filename] = []
                files_to_load[filename].append(key)

        # Weights 로드
        weights = {}
        for filename, keys in files_to_load.items():
            filepath = os.path.join(model_dir, filename)
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in keys:
                    weights[key] = f.get_tensor(key)

        # input_layernorm 구성 (RMSNorm)
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        ln_key = f"model.layers.{layer_idx}.input_layernorm.weight"
        self.input_layernorm.weight.data = weights[ln_key].float()
        self.input_layernorm = self.input_layernorm.to(self.device)

        # query_proj 구성 (2-layer MLP: Linear -> SiLU -> Linear, bias=False)
        # M+ 실제 코드: ACT2FN[self.config.hidden_act] = SiLU
        q0_weight = weights[f"model.layers.{layer_idx}.self_attn.query_proj.0.weight"]
        q2_weight = weights[f"model.layers.{layer_idx}.self_attn.query_proj.2.weight"]

        self.query_proj = nn.Sequential(
            nn.Linear(q0_weight.shape[1], q0_weight.shape[0], bias=False),
            nn.SiLU(),  # Llama uses SiLU (Swish), not ReLU
            nn.Linear(q2_weight.shape[1], q2_weight.shape[0], bias=False)
        )
        self.query_proj[0].weight.data = q0_weight.float()
        self.query_proj[2].weight.data = q2_weight.float()
        self.query_proj = self.query_proj.to(self.device)

        # key_proj 구성 (2-layer MLP, bias=False, SiLU)
        k0_weight = weights[f"model.layers.{layer_idx}.self_attn.key_proj.0.weight"]
        k2_weight = weights[f"model.layers.{layer_idx}.self_attn.key_proj.2.weight"]

        self.key_proj = nn.Sequential(
            nn.Linear(k0_weight.shape[1], k0_weight.shape[0], bias=False),
            nn.SiLU(),  # Llama uses SiLU (Swish), not ReLU
            nn.Linear(k2_weight.shape[1], k2_weight.shape[0], bias=False)
        )
        self.key_proj[0].weight.data = k0_weight.float()
        self.key_proj[2].weight.data = k2_weight.float()
        self.key_proj = self.key_proj.to(self.device)

    @torch.no_grad()
    def encode_tokens(self, text, proj_type="key", max_length=256):
        """
        텍스트 → token-level projected vectors (논문 방식)

        M+ 실제 코드 (line 2581):
            queries = query_proj(input_layernorm(hidden_states))  # [seq_len, 256]

        Returns:
            vectors: [num_tokens, selector_dim]
            mask: [num_tokens] - valid token mask
        """
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=max_length, padding=True
        ).to(self.device)

        # Llama forward → hidden states
        outputs = self.base_model.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )

        # layer_idx의 hidden states [batch=1, seq_len, hidden_size]
        hidden = outputs.hidden_states[self.layer_idx + 1]

        # 논문 방식: input_layernorm → projector (token-level)
        normed = self.input_layernorm(hidden.float())  # [1, seq_len, hidden_size]

        if proj_type == "key":
            vecs = self.key_proj(normed)  # [1, seq_len, selector_dim]
        else:
            vecs = self.query_proj(normed)

        # Remove batch dim, return with mask
        return vecs.squeeze(0).cpu().float(), inputs.attention_mask.squeeze(0).cpu()

    def encode_documents(self, texts, max_length=256):
        """
        여러 documents를 token-level key vectors로 인코딩

        Returns:
            List of (key_vectors, mask) for each document
        """
        doc_keys = []
        for text in tqdm(texts, desc="Encoding docs", leave=False):
            keys, mask = self.encode_tokens(text, "key", max_length)
            doc_keys.append((keys, mask))
        return doc_keys

    def retrieve_token_level(self, query_text, doc_keys_list, top_k=5):
        """
        Token-level 검색 (M+ 실제 방식)

        M+ 코드 (line 2581-2582):
            queries = query_proj(input_layernorm(hidden_states))  # [seq_len, 256]
            predictions = (queries @ ltm_keys.T).sigmoid().mean(dim=0)

        각 query 토큰이 모든 document의 key 토큰과 비교,
        document별로 평균 점수 계산
        """
        # Query encoding (token-level)
        query_vecs, query_mask = self.encode_tokens(query_text, "query")  # [q_len, 256]

        # Valid query tokens only
        valid_query = query_vecs[query_mask.bool()]  # [valid_q_len, 256]

        doc_scores = []
        for doc_keys, doc_mask in doc_keys_list:
            # Valid key tokens only
            valid_keys = doc_keys[doc_mask.bool()]  # [valid_k_len, 256]

            if len(valid_keys) == 0:
                doc_scores.append(0.0)
                continue

            # M+ 방식: sigmoid(query @ keys.T).mean()
            # [valid_q_len, 256] @ [256, valid_k_len] -> [valid_q_len, valid_k_len]
            similarities = torch.sigmoid(valid_query @ valid_keys.T)

            # 각 query 토큰에서 가장 높은 점수의 평균 (M+ style)
            # 또는 전체 평균
            score = similarities.mean().item()
            doc_scores.append(score)

        scores = torch.tensor(doc_scores)
        top_k_idx = torch.topk(scores, k=min(top_k, len(scores))).indices
        return top_k_idx.numpy(), scores[top_k_idx].numpy()


class LlamaRMSNorm(nn.Module):
    """Llama의 RMSNorm (LlamaRMSNorm과 동일)"""
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
    """HotpotQA에서 2-hop 샘플과 distractor 로드"""
    print(f"\n[Data] HotpotQA 로드 중...")

    # distractor 버전 사용 (논문과 동일)
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)

    samples = []
    all_distractors = []
    seen_distractors = set()

    for item in tqdm(dataset, desc="Processing", leave=False):
        if len(samples) >= num_samples and len(all_distractors) >= num_distractors:
            break

        supporting_titles = set(item['supporting_facts']['title'])

        # title -> text 매핑
        title_to_text = {}
        for title, sentences in zip(item['context']['title'], item['context']['sentences']):
            title_to_text[title] = ' '.join(sentences)

        # 2-hop pair 추출
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

        # Distractor 수집
        for title, text in title_to_text.items():
            if title not in supporting_titles and title not in seen_distractors:
                if len(text) > 100:  # 너무 짧은 것 제외
                    all_distractors.append(text)
                    seen_distractors.add(title)

    all_distractors = all_distractors[:num_distractors]

    print(f"[Data] {len(samples)} 2-hop samples, {len(all_distractors)} distractors")
    if samples:
        print(f"[Data] Sample Q: {samples[0]['question'][:60]}...")

    return samples, all_distractors


def run_coretrieval_experiment(retriever, samples, distractors, gap_sizes, top_k=5, verbose_first=True):
    """
    Co-retrieval 실험

    Memory 구성: [prefix 50] + [doc1] + [gap distractors] + [doc2] + [suffix 20]
    """
    print(f"\n{'='*70}")
    print(f"  Co-retrieval 실험 시작")
    print(f"{'='*70}")
    print(f"  Samples: {len(samples)}, Distractors: {len(distractors)}, Top-k: {top_k}")
    print(f"  Gap sizes: {gap_sizes}")
    print(f"  Memory 구성: [prefix 50] + [doc1] + [gap N] + [doc2] + [suffix 20]")

    results = {}
    detailed_logs = []  # 상세 로그 저장
    first_sample_logged = False

    for gap in gap_sizes:
        print(f"\n  [Gap {gap}] 실험 중...")

        co_count = 0
        single_count = 0
        none_count = 0

        for sample_idx, sample in enumerate(tqdm(samples, desc=f"Gap={gap}", leave=False)):
            # Memory 구성
            memory = []

            # Prefix distractors (50개)
            n_prefix = min(50, len(distractors))
            prefix_idx = np.random.choice(len(distractors), n_prefix, replace=False)
            memory.extend([distractors[i] for i in prefix_idx])

            # Doc1
            doc1_idx = len(memory)
            memory.append(sample['doc1'])

            # Gap distractors
            if gap > 0:
                n_gap = min(gap, len(distractors))
                gap_idx = np.random.choice(len(distractors), n_gap, replace=False)
                memory.extend([distractors[i] for i in gap_idx])

            # Doc2
            doc2_idx = len(memory)
            memory.append(sample['doc2'])

            # Suffix distractors (20개)
            n_suffix = min(20, len(distractors))
            suffix_idx = np.random.choice(len(distractors), n_suffix, replace=False)
            memory.extend([distractors[i] for i in suffix_idx])

            # 모든 document를 token-level key로 인코딩
            doc_keys = retriever.encode_documents(memory)

            # Question으로 token-level 검색
            top_k_idx, top_k_scores = retriever.retrieve_token_level(sample['question'], doc_keys, top_k)
            top_k_set = set(top_k_idx)

            # 결과 집계
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

            # 첫 번째 gap의 첫 번째 샘플에 대해 상세 로깅
            if verbose_first and not first_sample_logged and sample_idx == 0:
                first_sample_logged = True
                print(f"\n  {'─'*66}")
                print(f"  [상세 로그] 첫 번째 샘플 (Gap={gap})")
                print(f"  {'─'*66}")
                print(f"\n  ▶ INPUT")
                print(f"    Question: {sample['question']}")
                print(f"    Answer: {sample['answer']}")
                print(f"\n    Doc1 (title: {sample['title1']}):")
                print(f"      {sample['doc1'][:200]}...")
                print(f"      [총 {len(sample['doc1'])} chars]")
                print(f"\n    Doc2 (title: {sample['title2']}):")
                print(f"      {sample['doc2'][:200]}...")
                print(f"      [총 {len(sample['doc2'])} chars]")

                print(f"\n  ▶ MEMORY 구성")
                print(f"    Total documents: {len(memory)}")
                print(f"    - Prefix distractors: {n_prefix} (idx 0-{n_prefix-1})")
                print(f"    - Doc1 위치: idx {doc1_idx}")
                print(f"    - Gap distractors: {gap} (idx {doc1_idx+1}-{doc2_idx-1})" if gap > 0 else f"    - Gap distractors: 0")
                print(f"    - Doc2 위치: idx {doc2_idx}")
                print(f"    - Suffix distractors: {n_suffix} (idx {doc2_idx+1}-{len(memory)-1})")

                # 토큰 정보
                q_tokens = retriever.tokenizer.encode(sample['question'])
                d1_tokens = retriever.tokenizer.encode(sample['doc1'])
                d2_tokens = retriever.tokenizer.encode(sample['doc2'])
                print(f"\n  ▶ TOKEN 정보")
                print(f"    Question tokens: {len(q_tokens)}")
                print(f"    Doc1 tokens: {len(d1_tokens)}")
                print(f"    Doc2 tokens: {len(d2_tokens)}")
                print(f"    Avg distractor tokens: {np.mean([len(retriever.tokenizer.encode(d)) for d in distractors[:10]]):.1f}")

                print(f"\n  ▶ OUTPUT (검색 결과)")
                print(f"    Top-{top_k} retrieved indices: {list(top_k_idx)}")
                print(f"    Top-{top_k} scores: {[f'{s:.4f}' for s in top_k_scores]}")
                print(f"    Doc1 (idx {doc1_idx}) found: {'✓' if d1_found else '✗'}")
                print(f"    Doc2 (idx {doc2_idx}) found: {'✓' if d2_found else '✗'}")
                print(f"    Result: {result_type.upper()}")
                print(f"  {'─'*66}")

            # 상세 로그 저장 (첫 몇 개 샘플만)
            if sample_idx < 3:
                detailed_logs.append({
                    'gap': int(gap),
                    'sample_idx': int(sample_idx),
                    'question': sample['question'],
                    'doc1_title': sample['title1'],
                    'doc2_title': sample['title2'],
                    'memory_size': int(len(memory)),
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
            'counts': {
                'co': co_count,
                'single': single_count,
                'none': none_count,
                'total': total
            }
        }

        print(f"      Co-retrieval: {co_count}/{total} ({100*co_count/total:.1f}%)")
        print(f"      Single:       {single_count}/{total} ({100*single_count/total:.1f}%)")
        print(f"      None:         {none_count}/{total} ({100*none_count/total:.1f}%)")

    return results, detailed_logs


def visualize_results(results, output_dir, config):
    """결과 시각화 및 저장"""
    os.makedirs(output_dir, exist_ok=True)

    gaps = sorted(results.keys())
    co_rates = [results[g]['co_retrieval_rate'] * 100 for g in gaps]
    single_rates = [results[g]['single_rate'] * 100 for g in gaps]
    none_rates = [results[g]['none_rate'] * 100 for g in gaps]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Line plot
    ax1 = axes[0]
    ax1.plot(gaps, co_rates, 'g-o', linewidth=2, markersize=8, label='Co-retrieval (both)')
    ax1.plot(gaps, single_rates, 'orange', linestyle='--', marker='s', linewidth=2, markersize=8, label='Single (one only)')
    ax1.plot(gaps, none_rates, 'r:', marker='^', linewidth=2, markersize=8, label='None')
    ax1.set_xlabel('Gap Size (# distractors)', fontsize=12)
    ax1.set_ylabel('Rate (%)', fontsize=12)
    ax1.set_title('M+ Retriever: Co-retrieval vs Gap Size\n(Token-level, SiLU activation)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    for x, y in zip(gaps, co_rates):
        ax1.annotate(f'{y:.0f}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    # Stacked bar
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

    fig_path = os.path.join(output_dir, 'phase1_mplus_coretrieval.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # JSON 저장 (상세 정보 포함)
    from datetime import datetime
    json_path = os.path.join(output_dir, 'phase1_mplus_coretrieval.json')
    output_data = {
        'experiment': 'Phase 1: M+ Retriever Co-retrieval Analysis',
        'timestamp': datetime.now().isoformat(),
        'method': {
            'description': 'Token-level retrieval with M+ trained projectors',
            'activation': 'SiLU (matching Llama/M+ config)',
            'similarity': 'sigmoid(query_tokens @ key_tokens.T).mean()',
            'base_model': 'meta-llama/Llama-3.1-8B (4-bit)',
            'retriever_weights': 'YuWangX/mplus-8b'
        },
        'config': config,
        'results': {str(k): v for k, v in results.items()},
        'summary': {
            'gap_range': [min(gaps), max(gaps)],
            'co_retrieval_drop': results[min(gaps)]['co_retrieval_rate'] - results[max(gaps)]['co_retrieval_rate'] if len(gaps) >= 2 else 0
        }
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 상세 로그 별도 저장
    if 'detailed_logs' in config and config['detailed_logs']:
        logs_path = os.path.join(output_dir, 'phase1_mplus_detailed_logs.json')
        with open(logs_path, 'w') as f:
            json.dump(config['detailed_logs'], f, indent=2, ensure_ascii=False)

    print(f"\n  [저장된 파일]")
    print(f"    - 그래프: {os.path.abspath(fig_path)}")
    print(f"    - 결과:   {os.path.abspath(json_path)}")
    if 'detailed_logs' in config and config['detailed_logs']:
        print(f"    - 상세로그: {os.path.abspath(logs_path)}")


def print_summary(results, output_dir):
    """결과 요약 출력"""
    print(f"\n")
    print(f"{'='*70}")
    print(f"  Phase 1 실험 결과: M+ Retriever Co-retrieval Analysis")
    print(f"{'='*70}")
    print(f"  Method: Token-level retrieval with SiLU activation (논문 방식)")
    print(f"  Output: {os.path.abspath(output_dir)}")
    print(f"{'='*70}")

    print(f"\n  {'Gap':>6} | {'Co-retrieval':>14} | {'Single':>12} | {'None':>10}")
    print(f"  {'-'*50}")

    for gap in sorted(results.keys()):
        r = results[gap]
        co_pct = r['co_retrieval_rate']*100
        single_pct = r['single_rate']*100
        none_pct = r['none_rate']*100
        print(f"  {gap:>6} | {co_pct:>13.1f}% | {single_pct:>11.1f}% | {none_pct:>9.1f}%")

    print(f"  {'-'*50}")

    gaps = sorted(results.keys())
    if len(gaps) >= 2:
        first, last = gaps[0], gaps[-1]
        first_rate = results[first]['co_retrieval_rate']
        last_rate = results[last]['co_retrieval_rate']
        drop = first_rate - last_rate

        print(f"\n  [핵심 발견]")
        print(f"  Gap {first} → {last}:")
        print(f"    - Co-retrieval: {first_rate*100:.1f}% → {last_rate*100:.1f}%")
        if first_rate > 0:
            print(f"    - 하락폭: {drop*100:.1f}%p ({drop/first_rate*100:.0f}% 상대 감소)")

        print(f"\n  [시사점]")
        if drop > 0.1:
            print(f"    Gap이 커질수록 두 관련 문서를 함께 검색하기 어려워짐")
            print(f"    → M+ retriever의 co-retrieval 한계 확인")
            print(f"    → Write-time Linking 방식이 필요할 수 있음")
        else:
            print(f"    Gap 증가에도 co-retrieval rate 유지됨")
            print(f"    → M+ retriever가 robust한 성능을 보임")

    print(f"\n{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M+ Retriever Co-retrieval Experiment")
    parser.add_argument('--num_samples', type=int, default=100, help='Number of 2-hop samples')
    parser.add_argument('--num_distractors', type=int, default=1000, help='Number of distractors')
    parser.add_argument('--top_k', type=int, default=5, help='Top-k for retrieval')
    parser.add_argument('--layer_idx', type=int, default=16, help='Layer index (0-31)')
    parser.add_argument('--output_dir', type=str, default='./experiments/results_phase1_mplus', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Seed 설정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Quick test 모드
    if args.quick:
        args.num_samples = 30
        args.num_distractors = 500
        gap_sizes = [0, 50, 200]
    else:
        gap_sizes = [0, 10, 50, 100, 200, 500]

    print(f"\n{'='*60}")
    print("Phase 1: M+ Retriever Co-retrieval Experiment")
    print(f"{'='*60}")
    print(f"Samples: {args.num_samples}")
    print(f"Distractors: {args.num_distractors}")
    print(f"Top-k: {args.top_k}")
    print(f"Layer: {args.layer_idx}")
    print(f"Gaps: {gap_sizes}")

    # 데이터 로드
    samples, distractors = load_hotpotqa_data(args.num_samples, args.num_distractors)

    # Retriever 로드
    retriever = MPlusRetrieverProper(layer_idx=args.layer_idx)

    # 실험 실행
    results, detailed_logs = run_coretrieval_experiment(
        retriever, samples, distractors,
        gap_sizes=gap_sizes,
        top_k=args.top_k
    )

    # 실험 config (JSON 직렬화를 위해 native Python 타입으로 변환)
    config = {
        'num_samples': int(len(samples)),
        'num_distractors': int(len(distractors)),
        'top_k': int(args.top_k),
        'layer_idx': int(args.layer_idx),
        'gap_sizes': [int(g) for g in gap_sizes],
        'seed': int(args.seed),
        'quick_mode': bool(args.quick),
        'detailed_logs': detailed_logs  # 상세 로그 포함
    }

    # 결과 출력 및 저장
    print_summary(results, args.output_dir)
    visualize_results(results, args.output_dir, config)

    print(f"\n{'='*70}")
    print(f"  실험 완료!")
    print(f"  결과 디렉토리: {os.path.abspath(args.output_dir)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
