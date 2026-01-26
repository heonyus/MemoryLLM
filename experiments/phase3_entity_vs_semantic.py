#!/usr/bin/env python3
"""
Phase 3 (B-5): Entity vs Semantic Linking 전략 비교
====================================================
Write-time Linking에서 어떤 연결 기준이 더 효과적인가?

비교 전략:
1. Baseline: 링크 없음 (Phase 2 결과 재사용)
2. Semantic-only: 임베딩 유사도 기반 (Phase 2 Link-3 결과 재사용)
3. Entity-only: Named Entity 공유 기반 (새로 실행)
4. Hybrid-OR: Entity OR Semantic (새로 실행)
5. Hybrid-AND: Entity AND Semantic (새로 실행)

실행:
    python phase3_entity_vs_semantic.py --quick --encoder mplus
    python phase3_entity_vs_semantic.py --quick --encoder mplus --load_phase2 path/to/results.json
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import spacy
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# M+ Retriever Encoder (Phase 2와 동일)
# ============================================================

class MPlusEncoder:
    """M+ Retriever - Token-level retrieval 지원"""

    def __init__(self, model_id="YuWangX/mplus-8b", base_model_id="meta-llama/Llama-3.1-8B", layer_idx=16):
        print(f"\n  [M+ Encoder] 로드 중 (Token-level)...")
        self.layer_idx = layer_idx
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_mplus = True

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

        self._load_retriever_weights(model_id, layer_idx)
        print(f"  [M+ Encoder] 로드 완료 (GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB)")

    def _load_retriever_weights(self, model_id, layer_idx):
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
            f"model.layers.{layer_idx}.self_attn.key_proj.0.weight",
            f"model.layers.{layer_idx}.self_attn.key_proj.2.weight",
            f"model.layers.{layer_idx}.self_attn.query_proj.0.weight",
            f"model.layers.{layer_idx}.self_attn.query_proj.2.weight",
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

        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.input_layernorm.weight.data = weights[f"model.layers.{layer_idx}.input_layernorm.weight"].float()
        self.input_layernorm = self.input_layernorm.to(self.device)

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

    @torch.no_grad()
    def encode_tokens(self, text, proj_type="key", max_length=256):
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=max_length, padding=True
        ).to(self.device)

        outputs = self.base_model.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )

        hidden = outputs.hidden_states[self.layer_idx + 1]
        normed = self.input_layernorm(hidden.float())

        if proj_type == "key":
            vecs = self.key_proj(normed)
        else:
            vecs = self.query_proj(normed)

        return vecs.squeeze(0).cpu().float(), inputs.attention_mask.squeeze(0).cpu()

    def compute_token_level_score(self, query_tokens, query_mask, doc_tokens, doc_mask):
        valid_query = query_tokens[query_mask.bool()]
        valid_doc = doc_tokens[doc_mask.bool()]

        if len(valid_doc) == 0 or len(valid_query) == 0:
            return 0.0

        similarities = torch.sigmoid(valid_query @ valid_doc.T)
        return similarities.mean().item()


class LlamaRMSNorm(nn.Module):
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


# ============================================================
# Entity Extractor
# ============================================================

class EntityExtractor:
    """spaCy 기반 Named Entity 추출"""

    def __init__(self, model='en_core_web_sm'):
        print(f"  [NER] spaCy 모델 로드 중: {model}")
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"  [NER] 모델 다운로드 중...")
            os.system(f"python -m spacy download {model}")
            self.nlp = spacy.load(model)
        print(f"  [NER] 로드 완료")
        self._cache = {}

    def extract(self, text, doc_id=None):
        """텍스트에서 Named Entities 추출"""
        cache_key = doc_id if doc_id is not None else hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        doc = self.nlp(text)
        entities = set()
        for ent in doc.ents:
            # 정규화: 소문자, 공백 정리
            normalized = ent.text.lower().strip()
            if len(normalized) > 1:  # 너무 짧은 것 제외
                entities.add(normalized)

        self._cache[cache_key] = entities
        return entities

    def get_shared_entities(self, text1, text2, id1=None, id2=None):
        """두 문서 간 공유 엔티티 반환"""
        ent1 = self.extract(text1, id1)
        ent2 = self.extract(text2, id2)
        return ent1 & ent2


# ============================================================
# Linking Strategies
# ============================================================

class LinkingStrategy:
    """Base class for linking strategies"""
    def should_link(self, **kwargs):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__


class SemanticLinking(LinkingStrategy):
    """Semantic 유사도 기반 연결 (Phase 2와 동일)"""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def should_link(self, emb1, emb2, **kwargs):
        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0]
        return sim >= self.threshold, {'similarity': float(sim)}

    @property
    def name(self):
        return "Semantic"


class EntityLinking(LinkingStrategy):
    """Entity 공유 기반 연결"""

    def __init__(self, entity_extractor, min_shared=1):
        self.extractor = entity_extractor
        self.min_shared = min_shared

    def should_link(self, text1, text2, id1=None, id2=None, **kwargs):
        shared = self.extractor.get_shared_entities(text1, text2, id1, id2)
        return len(shared) >= self.min_shared, {'shared_entities': list(shared)}

    @property
    def name(self):
        return "Entity"


class HybridORLinking(LinkingStrategy):
    """Entity OR Semantic (둘 중 하나만 만족해도 연결)"""

    def __init__(self, entity_linker, semantic_linker):
        self.entity_linker = entity_linker
        self.semantic_linker = semantic_linker

    def should_link(self, **kwargs):
        entity_result, entity_info = self.entity_linker.should_link(**kwargs)
        semantic_result, semantic_info = self.semantic_linker.should_link(**kwargs)
        return entity_result or semantic_result, {
            'entity': entity_result,
            'semantic': semantic_result,
            **entity_info,
            **semantic_info
        }

    @property
    def name(self):
        return "Hybrid-OR"


class HybridANDLinking(LinkingStrategy):
    """Entity AND Semantic (둘 다 만족해야 연결)"""

    def __init__(self, entity_linker, semantic_linker):
        self.entity_linker = entity_linker
        self.semantic_linker = semantic_linker

    def should_link(self, **kwargs):
        entity_result, entity_info = self.entity_linker.should_link(**kwargs)
        semantic_result, semantic_info = self.semantic_linker.should_link(**kwargs)
        return entity_result and semantic_result, {
            'entity': entity_result,
            'semantic': semantic_result,
            **entity_info,
            **semantic_info
        }

    @property
    def name(self):
        return "Hybrid-AND"


# ============================================================
# Linked Memory with Strategy
# ============================================================

class LinkedMemoryWithStrategy:
    """
    전략 기반 Write-time Linking을 지원하는 메모리 시스템
    """

    def __init__(self, encoder, linking_strategy, link_k=3):
        self.encoder = encoder
        self.strategy = linking_strategy
        self.link_k = link_k
        self.is_mplus = getattr(encoder, 'is_mplus', False)

        # 메모리 저장소
        self.embeddings = []
        self.texts = []
        self.links = {}

        # M+ 전용
        if self.is_mplus:
            self.token_keys = []
        else:
            self.token_keys = None

        # 통계
        self.stats = {
            'total_links_created': 0,
            'links_per_document': [],
            'link_reasons': [],  # 링크 생성 이유 추적
        }

    def clear(self):
        self.embeddings = []
        self.texts = []
        self.links = {}
        if self.is_mplus:
            self.token_keys = []
        self.stats = {
            'total_links_created': 0,
            'links_per_document': [],
            'link_reasons': [],
        }

    def store(self, text, create_links=True):
        """문서 저장 (전략 기반 Write-time Linking)"""
        idx = len(self.embeddings)

        # Encoding
        if self.is_mplus:
            key_vecs, mask = self.encoder.encode_tokens(text, "key")
            self.token_keys.append((key_vecs, mask))
            mask_expanded = mask.unsqueeze(-1).float()
            emb = (key_vecs * mask_expanded).sum(dim=0) / mask_expanded.sum(dim=0)
            emb = emb.numpy()
        else:
            emb = self.encoder.encode(text, convert_to_numpy=True)
            if emb.ndim > 1:
                emb = emb.flatten()

        # Write-time Linking
        links_created = 0
        if create_links and self.link_k > 0 and len(self.embeddings) > 0:
            candidates = []

            for existing_idx in range(len(self.embeddings)):
                existing_text = self.texts[existing_idx]
                existing_emb = self.embeddings[existing_idx]

                should_link, info = self.strategy.should_link(
                    text1=text,
                    text2=existing_text,
                    id1=idx,
                    id2=existing_idx,
                    emb1=emb,
                    emb2=existing_emb,
                )

                if should_link:
                    # Semantic 유사도로 정렬용 점수 계산
                    sim = cosine_similarity(emb.reshape(1, -1), existing_emb.reshape(1, -1))[0, 0]
                    candidates.append((existing_idx, sim, info))

            # Top-k 링크만 유지 (semantic 유사도 순)
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:self.link_k]

            # 양방향 링크 생성
            self.links[idx] = []
            for linked_idx, sim, info in candidates:
                self.links[idx].append(linked_idx)
                if linked_idx not in self.links:
                    self.links[linked_idx] = []
                self.links[linked_idx].append(idx)
                links_created += 1
                self.stats['link_reasons'].append(info)

        # 저장
        self.embeddings.append(emb)
        self.texts.append(text)
        self.stats['total_links_created'] += links_created
        self.stats['links_per_document'].append(links_created)

        return idx

    def retrieve(self, query, top_k=5, expand_links=True, max_expansion=10):
        """검색 (Link Expansion 포함)"""
        if len(self.embeddings) == 0:
            return [], []

        if self.is_mplus:
            sims = self._retrieve_token_level(query)
        else:
            query_emb = self.encoder.encode(query, convert_to_numpy=True)
            if query_emb.ndim == 1:
                query_emb = query_emb.reshape(1, -1)
            all_embs = np.vstack(self.embeddings)
            sims = cosine_similarity(query_emb, all_embs)[0]

        top_k_idx = np.argsort(sims)[-top_k:][::-1]
        top_k_scores = sims[top_k_idx]

        if not expand_links:
            return list(top_k_idx), list(top_k_scores)

        # Link Expansion
        expanded = set(top_k_idx)
        expansion_scores = {int(i): float(sims[i]) for i in top_k_idx}

        for idx in list(top_k_idx):
            if int(idx) in self.links:
                for linked_idx in self.links[int(idx)]:
                    if linked_idx not in expanded:
                        expanded.add(linked_idx)
                        expansion_scores[linked_idx] = -1.0

        if len(expanded) > top_k + max_expansion:
            original = set(top_k_idx)
            extra = expanded - original
            extra_with_sims = [(i, sims[i]) for i in extra]
            extra_with_sims.sort(key=lambda x: x[1], reverse=True)
            extra = [i for i, _ in extra_with_sims[:max_expansion]]
            expanded = original | set(extra)

        result_indices = []
        result_scores = []
        for idx in expanded:
            result_indices.append(int(idx))
            result_scores.append(expansion_scores.get(int(idx), float(sims[idx])))

        return result_indices, result_scores

    def _retrieve_token_level(self, query):
        query_vecs, query_mask = self.encoder.encode_tokens(query, "query")
        scores = []
        for doc_keys, doc_mask in self.token_keys:
            score = self.encoder.compute_token_level_score(
                query_vecs, query_mask, doc_keys, doc_mask
            )
            scores.append(score)
        return np.array(scores)

    def get_link_stats(self):
        if not self.stats['links_per_document']:
            return {}
        return {
            'total_documents': len(self.embeddings),
            'total_links': self.stats['total_links_created'],
            'avg_links_per_doc': np.mean(self.stats['links_per_document']),
        }


# ============================================================
# Data Loading
# ============================================================

def load_hotpotqa_data(num_samples=100, num_distractors=500):
    """HotpotQA 데이터 로드"""
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


# ============================================================
# Experiment Runner
# ============================================================

def run_strategy_experiment(encoder, entity_extractor, samples, distractors, gap_sizes,
                            strategy, link_k=3, top_k=5):
    """단일 전략에 대한 실험 실행"""
    strategy_name = strategy.name if strategy else "Baseline"

    print(f"\n{'='*70}")
    print(f"  [{strategy_name}] Co-retrieval 실험")
    print(f"{'='*70}")
    print(f"  link_k={link_k}, top_k={top_k}")
    print(f"  Samples: {len(samples)}, Distractors: {len(distractors)}")

    results = {}

    for gap in gap_sizes:
        print(f"\n  [Gap {gap}] 실험 중...")

        co_count = 0
        single_count = 0
        none_count = 0
        doc1_doc2_link_count = 0

        for sample_idx, sample in enumerate(tqdm(samples, desc=f"Gap={gap}", leave=False)):
            # 메모리 초기화
            if strategy is None:
                # Baseline: 링크 없음
                memory = LinkedMemoryWithStrategy(encoder, SemanticLinking(), link_k=0)
            else:
                memory = LinkedMemoryWithStrategy(encoder, strategy, link_k=link_k)

            # Prefix distractors
            n_prefix = min(50, len(distractors))
            prefix_idx = np.random.choice(len(distractors), n_prefix, replace=False)
            for i in prefix_idx:
                memory.store(distractors[i], create_links=(strategy is not None))

            # Doc1
            doc1_idx = memory.store(sample['doc1'], create_links=(strategy is not None))

            # Gap distractors
            if gap > 0:
                n_gap = min(gap, len(distractors))
                gap_idx = np.random.choice(len(distractors), n_gap, replace=False)
                for i in gap_idx:
                    memory.store(distractors[i], create_links=(strategy is not None))

            # Doc2
            doc2_idx = memory.store(sample['doc2'], create_links=(strategy is not None))

            # Suffix distractors
            n_suffix = min(20, len(distractors))
            suffix_idx = np.random.choice(len(distractors), n_suffix, replace=False)
            for i in suffix_idx:
                memory.store(distractors[i], create_links=(strategy is not None))

            # Doc1-Doc2 링크 확인
            doc1_doc2_linked = (
                doc2_idx in memory.links.get(doc1_idx, []) or
                doc1_idx in memory.links.get(doc2_idx, [])
            )
            if doc1_doc2_linked:
                doc1_doc2_link_count += 1

            # 검색
            retrieved_idx, _ = memory.retrieve(
                sample['question'],
                top_k=top_k,
                expand_links=(strategy is not None)
            )
            retrieved_set = set(retrieved_idx)

            # 평가
            d1_found = doc1_idx in retrieved_set
            d2_found = doc2_idx in retrieved_set

            if d1_found and d2_found:
                co_count += 1
            elif d1_found or d2_found:
                single_count += 1
            else:
                none_count += 1

        total = len(samples)
        results[gap] = {
            'co_retrieval_rate': co_count / total,
            'single_rate': single_count / total,
            'none_rate': none_count / total,
            'doc1_doc2_link_rate': doc1_doc2_link_count / total,
            'counts': {
                'co': co_count,
                'single': single_count,
                'none': none_count,
                'doc1_doc2_linked': doc1_doc2_link_count,
                'total': total
            }
        }

        print(f"      Co-retrieval: {co_count}/{total} ({100*co_count/total:.1f}%)")
        print(f"      Doc1-Doc2 linked: {doc1_doc2_link_count}/{total} ({100*doc1_doc2_link_count/total:.1f}%)")

    return results


def run_all_strategies(encoder, entity_extractor, samples, distractors, gap_sizes,
                       link_k=3, top_k=5, phase2_results=None):
    """모든 전략 실행"""

    all_results = {}

    # 1. Baseline (Phase 2 결과 재사용 또는 새로 실행)
    if phase2_results and 'Baseline' in phase2_results:
        print("\n  [Baseline] Phase 2 결과 재사용")
        all_results['Baseline'] = phase2_results['Baseline']
    else:
        results = run_strategy_experiment(
            encoder, entity_extractor, samples, distractors, gap_sizes,
            strategy=None, link_k=0, top_k=top_k
        )
        all_results['Baseline'] = results

    # 2. Semantic-only (Phase 2 Link-3 결과 재사용 또는 새로 실행)
    if phase2_results and 'Link-3' in phase2_results:
        print("\n  [Semantic] Phase 2 Link-3 결과 재사용")
        all_results['Semantic'] = phase2_results['Link-3']
    else:
        semantic_strategy = SemanticLinking(threshold=0.0)
        results = run_strategy_experiment(
            encoder, entity_extractor, samples, distractors, gap_sizes,
            strategy=semantic_strategy, link_k=link_k, top_k=top_k
        )
        all_results['Semantic'] = results

    # 3. Entity-only (새로 실행)
    entity_strategy = EntityLinking(entity_extractor, min_shared=1)
    results = run_strategy_experiment(
        encoder, entity_extractor, samples, distractors, gap_sizes,
        strategy=entity_strategy, link_k=link_k, top_k=top_k
    )
    all_results['Entity'] = results

    # 4. Hybrid-OR (새로 실행)
    hybrid_or_strategy = HybridORLinking(
        EntityLinking(entity_extractor, min_shared=1),
        SemanticLinking(threshold=0.5)  # OR이므로 semantic threshold 높게
    )
    results = run_strategy_experiment(
        encoder, entity_extractor, samples, distractors, gap_sizes,
        strategy=hybrid_or_strategy, link_k=link_k, top_k=top_k
    )
    all_results['Hybrid-OR'] = results

    # 5. Hybrid-AND (새로 실행)
    hybrid_and_strategy = HybridANDLinking(
        EntityLinking(entity_extractor, min_shared=1),
        SemanticLinking(threshold=0.0)  # AND이므로 semantic threshold 낮게
    )
    results = run_strategy_experiment(
        encoder, entity_extractor, samples, distractors, gap_sizes,
        strategy=hybrid_and_strategy, link_k=link_k, top_k=top_k
    )
    all_results['Hybrid-AND'] = results

    return all_results


# ============================================================
# Visualization & Output
# ============================================================

def print_comparison_summary(all_results):
    """결과 요약 출력"""
    strategies = list(all_results.keys())
    gaps = sorted(all_results[strategies[0]].keys())

    print(f"\n")
    print(f"{'='*90}")
    print(f"  Phase 3 (B-5): Entity vs Semantic Linking 전략 비교")
    print(f"{'='*90}")

    # 헤더
    header = f"  {'Gap':>6} |"
    for strategy in strategies:
        header += f" {strategy:>12} |"
    print(header)
    print(f"  {'-'*(10 + 15*len(strategies))}")

    # 각 Gap별 결과
    for gap in gaps:
        row = f"  {gap:>6} |"
        for strategy in strategies:
            rate = all_results[strategy][gap]['co_retrieval_rate'] * 100
            row += f" {rate:>11.1f}% |"
        print(row)

    print(f"  {'-'*(10 + 15*len(strategies))}")

    # 개선 효과 분석
    print(f"\n  [Baseline 대비 개선]")
    if 'Baseline' in strategies:
        for gap in gaps:
            baseline = all_results['Baseline'][gap]['co_retrieval_rate'] * 100
            print(f"    Gap {gap}:")
            for strategy in strategies:
                if strategy != 'Baseline':
                    rate = all_results[strategy][gap]['co_retrieval_rate'] * 100
                    diff = rate - baseline
                    print(f"      {strategy}: {rate:.1f}% ({diff:+.1f}%p)")

    print(f"\n{'='*90}")


def visualize_results(all_results, output_dir, config):
    """결과 시각화"""
    os.makedirs(output_dir, exist_ok=True)

    strategies = list(all_results.keys())
    gaps = sorted(all_results[strategies[0]].keys())

    colors = {
        'Baseline': 'gray',
        'Semantic': 'blue',
        'Entity': 'green',
        'Hybrid-OR': 'orange',
        'Hybrid-AND': 'purple'
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Line plot
    ax1 = axes[0]
    for strategy in strategies:
        rates = [all_results[strategy][g]['co_retrieval_rate'] * 100 for g in gaps]
        ax1.plot(gaps, rates, '-o', label=strategy, color=colors.get(strategy, 'black'),
                 linewidth=2, markersize=8)
        for x, y in zip(gaps, rates):
            ax1.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=8)

    ax1.set_xlabel('Gap Size', fontsize=11)
    ax1.set_ylabel('Co-retrieval Rate (%)', fontsize=11)
    ax1.set_title('Entity vs Semantic Linking Strategies', fontsize=13)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Bar plot
    ax2 = axes[1]
    x = np.arange(len(gaps))
    width = 0.15

    for i, strategy in enumerate(strategies):
        rates = [all_results[strategy][g]['co_retrieval_rate'] * 100 for g in gaps]
        bars = ax2.bar(x + i * width, rates, width, label=strategy,
                       color=colors.get(strategy, 'black'), alpha=0.8)

    ax2.set_xlabel('Gap Size', fontsize=11)
    ax2.set_ylabel('Co-retrieval Rate (%)', fontsize=11)
    ax2.set_title('Comparison by Gap Size', fontsize=13)
    ax2.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax2.set_xticklabels([str(g) for g in gaps])
    ax2.legend()
    ax2.set_ylim([0, 105])

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'phase3_entity_vs_semantic.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # JSON 저장
    json_path = os.path.join(output_dir, 'phase3_entity_vs_semantic_results.json')
    output_data = {
        'experiment': 'Phase 3 (B-5): Entity vs Semantic Linking Strategies',
        'timestamp': datetime.now().isoformat(),
        'strategies': strategies,
        'config': config,
        'results': {
            strategy: {str(g): v for g, v in results.items()}
            for strategy, results in all_results.items()
        }
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n  [저장] {fig_path}")
    print(f"  [저장] {json_path}")

    return fig_path, json_path


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3: Entity vs Semantic Linking")
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_distractors', type=int, default=1000)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--link_k', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='./experiments/results_phase3')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--encoder', type=str, default='mplus', choices=['sbert', 'mplus'])
    parser.add_argument('--load_phase2', type=str, default=None,
                        help='Path to Phase 2 results JSON (to reuse Baseline & Semantic)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.quick:
        args.num_samples = 30
        args.num_distractors = 500
        gap_sizes = [0, 50, 200]
    else:
        gap_sizes = [0, 10, 50, 100, 200, 500]

    print(f"\n{'='*60}")
    print("Phase 3 (B-5): Entity vs Semantic Linking Strategies")
    print(f"{'='*60}")
    print(f"Encoder: {args.encoder.upper()}")
    print(f"Samples: {args.num_samples}, Distractors: {args.num_distractors}")
    print(f"Top-k: {args.top_k}, Link-k: {args.link_k}")
    print(f"Gaps: {gap_sizes}")

    # Phase 2 결과 로드 (있으면)
    phase2_results = None
    if args.load_phase2 and os.path.exists(args.load_phase2):
        print(f"\n[Phase 2] 결과 로드: {args.load_phase2}")
        with open(args.load_phase2) as f:
            phase2_data = json.load(f)
        # Convert string keys to int
        phase2_results = {}
        for cond, results in phase2_data.get('results', {}).items():
            phase2_results[cond] = {int(g): v for g, v in results.items()}
        print(f"  로드된 조건: {list(phase2_results.keys())}")

    # 데이터 로드
    samples, distractors = load_hotpotqa_data(args.num_samples, args.num_distractors)

    # Entity Extractor 로드
    entity_extractor = EntityExtractor()

    # Encoder 로드
    if args.encoder == 'mplus':
        print(f"\n[Encoder] M+ Retriever 로드 중...")
        encoder = MPlusEncoder()
    else:
        from sentence_transformers import SentenceTransformer
        print(f"\n[Encoder] SentenceTransformer 로드 중...")
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        encoder.is_mplus = False

    # 실험 실행
    all_results = run_all_strategies(
        encoder, entity_extractor, samples, distractors, gap_sizes,
        link_k=args.link_k, top_k=args.top_k,
        phase2_results=phase2_results
    )

    # Config
    config = {
        'encoder': args.encoder,
        'num_samples': int(len(samples)),
        'num_distractors': int(len(distractors)),
        'top_k': int(args.top_k),
        'link_k': int(args.link_k),
        'gap_sizes': [int(g) for g in gap_sizes],
        'seed': int(args.seed),
        'quick_mode': bool(args.quick),
        'phase2_loaded': args.load_phase2 is not None,
    }

    # 결과 출력 및 저장
    print_comparison_summary(all_results)
    visualize_results(all_results, args.output_dir, config)

    print(f"\n  실험 완료! 결과: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
