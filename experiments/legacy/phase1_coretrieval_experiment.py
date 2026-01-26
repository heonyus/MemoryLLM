#!/usr/bin/env python3
"""
Phase 1: M+ Retriever Co-retrieval 분석
========================================

목표: M+ retriever가 multi-hop에 필요한 co-retrieval을 잘 하는가?
가설: Gap(시간적 거리)이 커지면 co-retrieval rate 급감

사전 요구사항:
    python extract_mplus_retriever.py  # weights 추출 (한 번만)

실행:
    python phase1_coretrieval_experiment.py

    # 빠른 테스트 (샘플 50개, gap 3개만)
    python phase1_coretrieval_experiment.py --quick_test

결과물:
    results_phase1/
    ├── phase1_coretrieval_results.json
    └── phase1_coretrieval_results.png
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import gc

# ============================================================
# Configuration
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="M+ Retriever Co-retrieval Analysis")
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of 2-hop pairs to evaluate')
    parser.add_argument('--num_distractors', type=int, default=2000,
                        help='Number of distractor contexts')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top-k for retrieval')
    parser.add_argument('--layer_idx', type=int, default=16,
                        help='Layer index for retriever (0-31)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results_phase1',
                        help='Output directory')
    parser.add_argument('--weights_path', type=str, default='mplus_retriever_weights.pt',
                        help='Path to extracted retriever weights')
    parser.add_argument('--tokenizer_path', type=str, default='YuWangX/mplus-8b',
                        help='HuggingFace tokenizer path')
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test with fewer samples')
    parser.add_argument('--use_sentence_bert', action='store_true',
                        help='Use SentenceBERT as fallback if weights not found')
    return parser.parse_args()


def seed_everything(seed):
    """재현성을 위한 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def print_gpu_memory():
    """GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"    GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


# ============================================================
# M+ Retriever (Lightweight)
# ============================================================

class MPlusRetrieverLightweight(nn.Module):
    """
    M+ Retriever 경량 버전
    
    추출된 weights 파일 사용하여 전체 모델 로드 없이 작동
    """
    
    def __init__(self, weights_path, tokenizer_path, layer_idx=16, device='cuda'):
        super().__init__()
        self.device = device
        self.layer_idx = layer_idx
        
        print(f"\n[Retriever] M+ Retriever 로드 중...")
        print(f"    Weights: {weights_path}")
        print(f"    Layer: {layer_idx}")
        
        # 1. Weights 로드
        weights = torch.load(weights_path, map_location='cpu')
        config = weights['config']
        
        self.hidden_size = config['hidden_size']
        self.selector_dim = config['selector_hidden_dim']
        self.vocab_size = config['vocab_size']
        
        print(f"    Hidden size: {self.hidden_size}")
        print(f"    Selector dim: {self.selector_dim}")
        
        # 2. Embedding layer 복원
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embed_tokens.load_state_dict(weights['embed_tokens'])
        self.embed_tokens = self.embed_tokens.to(device).half()
        
        # 3. Layer-specific components 복원
        layer_weights = weights['layers'][layer_idx]
        
        # Input LayerNorm
        self.input_layernorm = nn.LayerNorm(self.hidden_size)
        self.input_layernorm.load_state_dict(layer_weights['input_layernorm'])
        self.input_layernorm = self.input_layernorm.to(device).half()
        
        # Query Projector (2-layer MLP)
        self.query_proj = self._build_mlp(layer_weights['query_proj'])
        self.query_proj = self.query_proj.to(device).half()
        
        # Key Projector (2-layer MLP)
        self.key_proj = self._build_mlp(layer_weights['key_proj'])
        self.key_proj = self.key_proj.to(device).half()
        
        # 4. Tokenizer 로드
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        print(f"    ✓ M+ Retriever 로드 완료")
        print_gpu_memory()
        
        # 메모리 정리
        del weights
        gc.collect()
    
    def _build_mlp(self, state_dict):
        """State dict에서 2-layer MLP 복원"""
        # State dict에서 차원 추론
        # 일반적으로: 0.weight, 0.bias, 2.weight, 2.bias (0=Linear, 1=ReLU, 2=Linear)
        if '0.weight' in state_dict:
            in_features = state_dict['0.weight'].shape[1]
            hidden_features = state_dict['0.weight'].shape[0]
            out_features = state_dict['2.weight'].shape[0]
            
            mlp = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, out_features)
            )
        else:
            # 다른 형식인 경우 처리
            keys = list(state_dict.keys())
            raise ValueError(f"Unexpected state_dict format: {keys}")
        
        mlp.load_state_dict(state_dict)
        return mlp
    
    @torch.no_grad()
    def encode_texts(self, texts, batch_size=32, show_progress=True):
        """텍스트들을 key vectors로 인코딩"""
        if isinstance(texts, str):
            texts = [texts]
        
        all_keys = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts", leave=False)
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Embedding
            hidden_states = self.embed_tokens(input_ids)  # [B, seq, hidden]
            
            # Mean pooling
            mask = attention_mask.unsqueeze(-1).half()
            pooled = (hidden_states * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            
            # Key projection
            normed = self.input_layernorm(pooled)
            keys = self.key_proj(normed)  # [B, selector_dim]
            
            all_keys.append(keys.cpu())
        
        return torch.cat(all_keys, dim=0)
    
    @torch.no_grad()
    def compute_query(self, text):
        """Query 텍스트를 query vector로 변환"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        hidden_states = self.embed_tokens(input_ids)
        mask = attention_mask.unsqueeze(-1).half()
        pooled = (hidden_states * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        
        normed = self.input_layernorm(pooled)
        query = self.query_proj(normed)
        
        return query
    
    def retrieve(self, query_text, key_vectors, k=5):
        """M+ 스타일 검색"""
        query_vec = self.compute_query(query_text)  # [1, selector_dim]
        keys = key_vectors.to(self.device).half()
        
        # M+ style: sigmoid of dot product
        scores = (query_vec @ keys.T).sigmoid().squeeze(0)
        
        top_k = torch.topk(scores, k=min(k, len(scores)))
        return top_k.indices.cpu().numpy(), top_k.values.cpu().numpy()


class SentenceBERTRetriever:
    """
    Fallback: SentenceBERT 기반 Retriever
    M+ weights가 없을 때 사용
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        print(f"\n[Retriever] SentenceBERT 로드 중... (fallback)")
        print(f"    Model: {model_name}")
        
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        
        print(f"    ✓ SentenceBERT 로드 완료")
        print_gpu_memory()
    
    def encode_texts(self, texts, batch_size=32, show_progress=True):
        """텍스트 인코딩"""
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress,
            convert_to_tensor=True
        )
        return embeddings.cpu()
    
    def retrieve(self, query_text, key_vectors, k=5):
        """코사인 유사도 기반 검색"""
        query_vec = self.model.encode([query_text], convert_to_tensor=True)
        keys = key_vectors.to(self.device)
        
        # Cosine similarity
        query_norm = query_vec / query_vec.norm(dim=-1, keepdim=True)
        keys_norm = keys / keys.norm(dim=-1, keepdim=True)
        scores = (query_norm @ keys_norm.T).squeeze(0)
        
        top_k = torch.topk(scores, k=min(k, len(scores)))
        return top_k.indices.cpu().numpy(), top_k.values.cpu().numpy()


def load_retriever(args):
    """Retriever 로드 (M+ 또는 fallback)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.use_sentence_bert:
        return SentenceBERTRetriever(device=device)
    
    if os.path.exists(args.weights_path):
        try:
            return MPlusRetrieverLightweight(
                args.weights_path, 
                args.tokenizer_path,
                layer_idx=args.layer_idx,
                device=device
            )
        except Exception as e:
            print(f"\n    ⚠️ M+ Retriever 로드 실패: {e}")
            print(f"    SentenceBERT fallback 사용...")
            return SentenceBERTRetriever(device=device)
    else:
        print(f"\n    ⚠️ Weights 파일 없음: {args.weights_path}")
        print(f"    먼저 extract_mplus_retriever.py를 실행하거나,")
        print(f"    --use_sentence_bert 플래그를 사용하세요.")
        print(f"    SentenceBERT fallback 사용...")
        return SentenceBERTRetriever(device=device)


# ============================================================
# HotpotQA 데이터 로드
# ============================================================

def load_hotpotqa_data(num_pairs=200, num_distractors=2000):
    """HotpotQA에서 2-hop pairs와 distractors 추출"""
    print(f"\n[Data] HotpotQA 로드 중...")
    
    from datasets import load_dataset
    dataset = load_dataset('hotpot_qa', 'distractor', split='validation', trust_remote_code=True)
    print(f"    ✓ Validation: {len(dataset):,}개 로드")
    
    twohop_pairs = []
    distractors = []
    seen_distractors = set()
    
    for item in tqdm(dataset, desc="Processing", leave=False):
        sup_titles = list(set(item['supporting_facts']['title']))
        
        if len(sup_titles) >= 2:
            title_to_text = {}
            for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                title_to_text[title] = ' '.join(sentences)
            
            if sup_titles[0] in title_to_text and sup_titles[1] in title_to_text:
                twohop_pairs.append({
                    'context1': title_to_text[sup_titles[0]],
                    'context2': title_to_text[sup_titles[1]],
                    'question': item['question'],
                    'answer': item['answer'],
                    'title1': sup_titles[0],
                    'title2': sup_titles[1]
                })
            
            for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                if title not in sup_titles and title not in seen_distractors:
                    text = ' '.join(sentences)
                    if len(text) > 100:
                        distractors.append(text)
                        seen_distractors.add(title)
        
        if len(twohop_pairs) >= num_pairs and len(distractors) >= num_distractors:
            break
    
    twohop_pairs = twohop_pairs[:num_pairs]
    distractors = distractors[:num_distractors]
    
    print(f"    ✓ 2-hop pairs: {len(twohop_pairs)}개")
    print(f"    ✓ Distractors: {len(distractors)}개")
    
    if twohop_pairs:
        sample = twohop_pairs[0]
        print(f"\n    [Sample]")
        print(f"    Q: {sample['question'][:80]}...")
        print(f"    A: {sample['answer']}")
    
    return twohop_pairs, distractors


# ============================================================
# Co-retrieval 실험
# ============================================================

def run_coretrieval_experiment(retriever, twohop_pairs, distractors, distractor_keys,
                                gap_sizes, top_k=5):
    """
    Co-retrieval 실험
    
    메모리 구성:
    [Pre-dist 50개] → [Doc1] → [Gap-dist N개] → [Doc2] → [Post-dist 20개]
    """
    print(f"\n[Experiment] Co-retrieval 실험 시작")
    print(f"    Gap sizes: {gap_sizes}")
    print(f"    Top-k: {top_k}")
    print(f"    Samples: {len(twohop_pairs)}")
    
    results = {}
    
    for gap in gap_sizes:
        print(f"\n    [Gap={gap}]")
        
        coretrieval_count = 0
        single_count = 0
        none_count = 0
        
        for pair in tqdm(twohop_pairs, desc=f"Gap={gap}", leave=False):
            # Context 인코딩
            ctx1_key = retriever.encode_texts([pair['context1']], show_progress=False)
            ctx2_key = retriever.encode_texts([pair['context2']], show_progress=False)
            
            # 메모리 구성
            n_dist = len(distractors)
            pre_dist_idx = random.sample(range(n_dist), min(50, n_dist))
            gap_dist_idx = random.sample(range(n_dist), min(gap, n_dist)) if gap > 0 else []
            post_dist_idx = random.sample(range(n_dist), min(20, n_dist))
            
            # Keys 조합
            keys_list = []
            
            if pre_dist_idx:
                keys_list.append(distractor_keys[pre_dist_idx])
            
            ctx1_idx = sum(len(k) for k in keys_list)
            keys_list.append(ctx1_key)
            
            if gap_dist_idx:
                keys_list.append(distractor_keys[gap_dist_idx])
            
            ctx2_idx = sum(len(k) for k in keys_list)
            keys_list.append(ctx2_key)
            
            if post_dist_idx:
                keys_list.append(distractor_keys[post_dist_idx])
            
            all_keys = torch.cat(keys_list, dim=0)
            
            # 검색
            indices, _ = retriever.retrieve(pair['question'], all_keys, k=top_k)
            indices_set = set(indices)
            
            ctx1_found = ctx1_idx in indices_set
            ctx2_found = ctx2_idx in indices_set
            
            if ctx1_found and ctx2_found:
                coretrieval_count += 1
            elif ctx1_found or ctx2_found:
                single_count += 1
            else:
                none_count += 1
        
        n = len(twohop_pairs)
        results[gap] = {
            'coretrieval_rate': coretrieval_count / n,
            'single_rate': single_count / n,
            'none_rate': none_count / n,
            'coretrieval_count': coretrieval_count,
            'single_count': single_count,
            'none_count': none_count,
            'total': n
        }
        
        print(f"      Co-retrieval: {coretrieval_count}/{n} = {coretrieval_count/n:.1%}")
        print(f"      Single:       {single_count}/{n} = {single_count/n:.1%}")
        print(f"      None:         {none_count}/{n} = {none_count/n:.1%}")
    
    return results


# ============================================================
# 결과 시각화 및 저장
# ============================================================

def visualize_results(results, output_dir):
    """결과 시각화"""
    print(f"\n[Visualize] 그래프 생성 중...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    gaps = sorted(results.keys())
    coretrieval_rates = [results[g]['coretrieval_rate'] for g in gaps]
    single_rates = [results[g]['single_rate'] for g in gaps]
    none_rates = [results[g]['none_rate'] for g in gaps]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Line plot
    ax1 = axes[0]
    ax1.plot(gaps, coretrieval_rates, 'o-', lw=2, ms=8, color='green', label='Co-retrieval (both)')
    ax1.plot(gaps, single_rates, 's--', lw=2, ms=8, color='orange', label='Single (one only)')
    ax1.plot(gaps, none_rates, '^:', lw=2, ms=8, color='red', label='None')
    
    ax1.set_xlabel('Gap Size (distractors between contexts)', fontsize=12)
    ax1.set_ylabel('Rate', fontsize=12)
    ax1.set_title('M+ Retriever: Co-retrieval vs Gap Size', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    for x, y in zip(gaps, coretrieval_rates):
        ax1.annotate(f'{y:.0%}', (x, y), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=9)
    
    # Stacked bar
    ax2 = axes[1]
    x_pos = np.arange(len(gaps))
    
    ax2.bar(x_pos, coretrieval_rates, 0.6, label='Co-retrieval', color='green', alpha=0.8)
    ax2.bar(x_pos, single_rates, 0.6, bottom=coretrieval_rates, 
            label='Single', color='orange', alpha=0.8)
    ax2.bar(x_pos, none_rates, 0.6, 
            bottom=[c+s for c, s in zip(coretrieval_rates, single_rates)],
            label='None', color='red', alpha=0.8)
    
    ax2.set_xlabel('Gap Size', fontsize=12)
    ax2.set_ylabel('Rate', fontsize=12)
    ax2.set_title('Retrieval Outcome Distribution', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(gaps)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'phase1_coretrieval_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 저장: {fig_path}")
    
    return fig_path


def save_results(results, args, output_dir):
    """JSON 저장"""
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'Phase 1 - M+ Retriever Co-retrieval Analysis',
        'config': {
            'weights_path': args.weights_path,
            'layer_idx': args.layer_idx,
            'top_k': args.top_k,
            'num_samples': args.num_samples,
            'num_distractors': args.num_distractors,
            'seed': args.seed,
            'use_sentence_bert': args.use_sentence_bert
        },
        'results': {str(k): v for k, v in results.items()},
    }
    
    json_path = os.path.join(output_dir, 'phase1_coretrieval_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"    ✓ 저장: {json_path}")
    
    return json_path


def print_summary(results):
    """결과 요약"""
    print("\n" + "=" * 60)
    print("📊 실험 결과 요약")
    print("=" * 60)
    
    print(f"\n{'Gap':>8} | {'Co-retrieval':>12} | {'Single':>10} | {'None':>8}")
    print("-" * 45)
    
    for gap in sorted(results.keys()):
        r = results[gap]
        print(f"{gap:>8} | {r['coretrieval_rate']:>11.1%} | {r['single_rate']:>9.1%} | {r['none_rate']:>7.1%}")
    
    print("-" * 45)
    
    gaps = sorted(results.keys())
    if len(gaps) >= 2:
        first, last = gaps[0], gaps[-1]
        drop = results[first]['coretrieval_rate'] - results[last]['coretrieval_rate']
        
        print(f"\n💡 핵심 발견:")
        print(f"   • Gap {first} → {last}: Co-retrieval {results[first]['coretrieval_rate']:.1%} → {results[last]['coretrieval_rate']:.1%}")
        print(f"   • 하락폭: {drop:.1%}p ({drop/results[first]['coretrieval_rate']*100:.0f}% 감소)")
        print(f"\n🎯 시사점:")
        print(f"   Gap이 커지면 두 관련 문서를 함께 검색하기 어려워짐")
        print(f"   → Write-time Linking이 필요!")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    
    if args.quick_test:
        args.num_samples = 50
        args.num_distractors = 500
    
    print("=" * 60)
    print("🔬 Phase 1: M+ Retriever Co-retrieval Analysis")
    print("=" * 60)
    print(f"Samples: {args.num_samples}")
    print(f"Distractors: {args.num_distractors}")
    print(f"Top-k: {args.top_k}")
    print(f"Seed: {args.seed}")
    
    seed_everything(args.seed)
    
    # Retriever 로드
    retriever = load_retriever(args)
    
    # 데이터 로드
    twohop_pairs, distractors = load_hotpotqa_data(
        num_pairs=args.num_samples,
        num_distractors=args.num_distractors
    )
    
    # Distractor 인코딩
    print(f"\n[Encoding] Distractor 인코딩 중...")
    distractor_keys = retriever.encode_texts(distractors, batch_size=args.batch_size)
    print(f"    ✓ 완료: {distractor_keys.shape}")
    
    # 실험
    if args.quick_test:
        gap_sizes = [0, 10, 20]
    else:
        gap_sizes = [0, 10, 50, 100, 200, 500, 1000]
    
    results = run_coretrieval_experiment(
        retriever, twohop_pairs, distractors, distractor_keys,
        gap_sizes=gap_sizes, top_k=args.top_k
    )
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    visualize_results(results, args.output_dir)
    save_results(results, args, args.output_dir)
    
    # 요약
    print_summary(results)
    
    print("\n" + "=" * 60)
    print(f"✅ 완료! 결과: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
