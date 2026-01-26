#!/usr/bin/env python3
"""
Phase 2 (B-3): Explicit Linking 실험
=====================================
Write-time Linking + Read-time Link Expansion이 co-retrieval을 개선하는지 검증

실험 설계:
- 저장 시: 새 문서와 기존 메모리 중 유사한 k개를 찾아 양방향 링크 생성
- 검색 시: Top-k 검색 후, 링크를 따라 연결된 문서들도 함께 반환

비교 조건:
- Baseline: 링크 없음
- Link-1: k=1
- Link-3: k=3
- Link-5: k=5

Retriever 비교:
- SBERT: sentence-transformers/all-MiniLM-L6-v2
- M+: YuWangX/mplus-8b의 retriever (Base Llama hidden states 사용)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# M+ Retriever Encoder (Token-level, 논문과 동일한 방식)
# ============================================================

class MPlusEncoder:
    """
    M+ Retriever - Token-level retrieval 지원

    논문 방식:
        keys = key_proj(input_layernorm(hidden_states))   # 저장 시
        queries = query_proj(input_layernorm(hidden_states))  # 검색 시
        scores = sigmoid(queries @ keys.T).mean()  # token-to-token 매칭
    """

    def __init__(self, model_id="YuWangX/mplus-8b", base_model_id="meta-llama/Llama-3.1-8B", layer_idx=16):
        print(f"\n  [M+ Encoder] 로드 중 (Token-level)...")
        self.layer_idx = layer_idx
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_mplus = True  # M+ encoder임을 표시

        # Base Llama 모델 로드 (4-bit)
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

        # M+ retriever weights 로드 (key_proj + query_proj)
        self._load_retriever_weights(model_id, layer_idx)
        print(f"  [M+ Encoder] 로드 완료 (GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB)")

    def _load_retriever_weights(self, model_id, layer_idx):
        """safetensors에서 retriever weights 로드 (key_proj + query_proj)"""
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

        # key_proj + query_proj 둘 다 로드
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

        # RMSNorm
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.input_layernorm.weight.data = weights[f"model.layers.{layer_idx}.input_layernorm.weight"].float()
        self.input_layernorm = self.input_layernorm.to(self.device)

        # key_proj (문서 인코딩용)
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

        # query_proj (쿼리 인코딩용)
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
        """
        텍스트 → token-level vectors (논문 방식)

        Returns:
            vectors: [num_tokens, 256] tensor
            mask: [num_tokens] attention mask
        """
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

    @torch.no_grad()
    def encode(self, text, convert_to_numpy=True, max_length=256):
        """
        텍스트 → document embedding (mean pooling)
        Write-time linking에서 유사도 계산용
        """
        if isinstance(text, list):
            return np.vstack([self.encode(t, convert_to_numpy) for t in text])

        keys, mask = self.encode_tokens(text, "key", max_length)

        # Mean pooling
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (keys * mask_expanded).sum(dim=0) / mask_expanded.sum(dim=0)

        if convert_to_numpy:
            return pooled.numpy()
        return pooled

    def compute_token_level_score(self, query_tokens, query_mask, doc_tokens, doc_mask):
        """
        Token-level 유사도 점수 계산 (M+ 논문 방식)

        score = sigmoid(query_tokens @ doc_tokens.T).mean()
        """
        # Valid tokens only
        valid_query = query_tokens[query_mask.bool()]  # [q_len, 256]
        valid_doc = doc_tokens[doc_mask.bool()]        # [d_len, 256]

        if len(valid_doc) == 0 or len(valid_query) == 0:
            return 0.0

        # sigmoid(q @ k.T).mean()
        similarities = torch.sigmoid(valid_query @ valid_doc.T)
        return similarities.mean().item()


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


class LinkedMemory:
    """
    Write-time Linking을 지원하는 메모리 시스템

    기존 방식:
        store(doc) → 그냥 저장
        retrieve(query) → Top-k 반환

    제안 방식:
        store(doc) → 기존 메모리에서 유사한 k개 찾아 링크 생성 후 저장
        retrieve(query) → Top-k + Link Expansion

    M+ 모드:
        - 저장: token-level key vectors 저장
        - 링크: document-level similarity (mean pooling)로 결정
        - 검색: token-level retrieval (sigmoid(q·k).mean())
    """

    def __init__(self, encoder, link_k=3, similarity_threshold=0.0):
        """
        Args:
            encoder: 텍스트 인코더 (SentenceTransformer 또는 MPlusEncoder)
            link_k: Write-time에 연결할 문서 수
            similarity_threshold: 링크 생성을 위한 최소 유사도 (0이면 항상 링크)
        """
        self.encoder = encoder
        self.link_k = link_k
        self.similarity_threshold = similarity_threshold

        # M+ encoder 여부 확인
        self.is_mplus = getattr(encoder, 'is_mplus', False)

        # 메모리 저장소
        self.embeddings = []       # List of numpy arrays (document-level, for linking)
        self.texts = []            # List of strings
        self.links = {}            # idx -> [linked_idx, ...]

        # M+ 전용: token-level 저장소
        if self.is_mplus:
            self.token_keys = []   # List of (key_vectors, mask) tuples
        else:
            self.token_keys = None

        # 통계 (분석용)
        self.stats = {
            'total_links_created': 0,
            'links_per_document': [],
        }

    def clear(self):
        """메모리 초기화"""
        self.embeddings = []
        self.texts = []
        self.links = {}
        if self.is_mplus:
            self.token_keys = []
        self.stats = {'total_links_created': 0, 'links_per_document': []}

    def store(self, text, create_links=True):
        """
        문서 저장 (Write-time Linking 포함)

        Args:
            text: 저장할 문서
            create_links: 링크 생성 여부 (False면 baseline과 동일)

        Returns:
            idx: 저장된 문서의 인덱스
        """
        idx = len(self.embeddings)

        # M+ 모드: token-level key vectors도 저장
        if self.is_mplus:
            key_vecs, mask = self.encoder.encode_tokens(text, "key")
            self.token_keys.append((key_vecs, mask))

            # Document embedding (linking용) - mean pooling
            mask_expanded = mask.unsqueeze(-1).float()
            emb = (key_vecs * mask_expanded).sum(dim=0) / mask_expanded.sum(dim=0)
            emb = emb.numpy().reshape(1, -1)
        else:
            # SBERT 모드: document embedding
            emb = self.encoder.encode(text, convert_to_numpy=True)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)

        # Write-time Retrieval & Linking (document-level similarity 사용)
        links_created = 0
        if create_links and self.link_k > 0 and len(self.embeddings) > 0:
            # 기존 메모리와 유사도 계산
            existing_embs = np.vstack(self.embeddings)
            sims = cosine_similarity(emb, existing_embs)[0]

            # Top-k 선택 (threshold 이상인 것만)
            top_k_idx = np.argsort(sims)[-self.link_k:][::-1]

            # 양방향 링크 생성
            self.links[idx] = []
            for linked_idx in top_k_idx:
                if sims[linked_idx] >= self.similarity_threshold:
                    # 새 문서 → 기존 문서 링크
                    self.links[idx].append(int(linked_idx))

                    # 기존 문서 → 새 문서 링크 (양방향)
                    if linked_idx not in self.links:
                        self.links[int(linked_idx)] = []
                    self.links[int(linked_idx)].append(idx)

                    links_created += 1

        # 저장
        self.embeddings.append(emb.flatten())
        self.texts.append(text)

        # 통계 업데이트
        self.stats['total_links_created'] += links_created
        self.stats['links_per_document'].append(links_created)

        return idx

    def retrieve(self, query, top_k=5, expand_links=True, max_expansion=10):
        """
        문서 검색 (Link Expansion 포함)

        M+ 모드: Token-level retrieval (sigmoid(q·k).mean())
        SBERT 모드: Document-level retrieval (cosine similarity)

        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            expand_links: 링크 확장 여부 (False면 baseline과 동일)
            max_expansion: 링크로 추가할 최대 문서 수

        Returns:
            indices: 검색된 문서 인덱스 리스트
            scores: 유사도 점수 (링크로 추가된 문서는 -1)
        """
        if len(self.embeddings) == 0:
            return [], []

        if self.is_mplus:
            # M+ Token-level Retrieval
            sims = self._retrieve_token_level(query)
        else:
            # SBERT Document-level Retrieval
            query_emb = self.encoder.encode(query, convert_to_numpy=True)
            if query_emb.ndim == 1:
                query_emb = query_emb.reshape(1, -1)
            all_embs = np.vstack(self.embeddings)
            sims = cosine_similarity(query_emb, all_embs)[0]

        # Top-k 선택
        top_k_idx = np.argsort(sims)[-top_k:][::-1]
        top_k_scores = sims[top_k_idx]

        if not expand_links:
            return list(top_k_idx), list(top_k_scores)

        # Link Expansion
        expanded = set(top_k_idx)
        expansion_scores = {int(idx): float(sims[idx]) for idx in top_k_idx}

        # 검색된 문서들의 링크를 따라감
        for idx in list(top_k_idx):
            if int(idx) in self.links:
                for linked_idx in self.links[int(idx)]:
                    if linked_idx not in expanded:
                        expanded.add(linked_idx)
                        expansion_scores[linked_idx] = -1.0  # 링크로 추가됨 표시

        # 최대 확장 수 제한
        if len(expanded) > top_k + max_expansion:
            # 원래 top_k는 유지하고, 나머지는 유사도 순으로 제한
            original = set(top_k_idx)
            extra = expanded - original
            extra_with_sims = [(idx, sims[idx]) for idx in extra]
            extra_with_sims.sort(key=lambda x: x[1], reverse=True)
            extra = [idx for idx, _ in extra_with_sims[:max_expansion]]
            expanded = original | set(extra)

        # 결과 정리
        result_indices = []
        result_scores = []
        for idx in expanded:
            result_indices.append(int(idx))
            result_scores.append(expansion_scores.get(int(idx), float(sims[idx])))

        return result_indices, result_scores

    def _retrieve_token_level(self, query):
        """
        M+ Token-level Retrieval

        논문 방식: score = sigmoid(query_tokens @ doc_keys.T).mean()
        """
        # Query를 token-level로 인코딩
        query_vecs, query_mask = self.encoder.encode_tokens(query, "query")

        # 각 문서와의 유사도 계산
        scores = []
        for doc_keys, doc_mask in self.token_keys:
            score = self.encoder.compute_token_level_score(
                query_vecs, query_mask, doc_keys, doc_mask
            )
            scores.append(score)

        return np.array(scores)

    def get_link_stats(self):
        """링크 통계 반환"""
        if not self.stats['links_per_document']:
            return {}
        return {
            'total_documents': len(self.embeddings),
            'total_links': self.stats['total_links_created'],
            'avg_links_per_doc': np.mean(self.stats['links_per_document']),
            'max_links_per_doc': max(self.stats['links_per_document']) if self.stats['links_per_document'] else 0,
            'documents_with_links': sum(1 for x in self.stats['links_per_document'] if x > 0),
        }


def load_hotpotqa_data(num_samples=100, num_distractors=500):
    """HotpotQA 데이터 로드 (Phase 1과 동일)"""
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


def run_linking_experiment(encoder, samples, distractors, gap_sizes,
                           link_k=3, top_k=5, expand_links=True,
                           verbose_first=True):
    """
    Explicit Linking 실험 실행

    핵심 차이점:
    - Phase 1: 모든 문서를 한번에 인코딩 후 검색
    - Phase 2: 문서를 순차적으로 저장하며 링크 생성

    Args:
        encoder: SentenceTransformer 인코더
        samples: HotpotQA 2-hop 샘플
        distractors: Distractor 문서들
        gap_sizes: 테스트할 gap 크기들
        link_k: Write-time에 연결할 문서 수 (0이면 baseline)
        top_k: Read-time에 검색할 문서 수
        expand_links: Link expansion 사용 여부
        verbose_first: 첫 샘플 상세 로깅
    """
    condition_name = f"Link-{link_k}" if link_k > 0 and expand_links else "Baseline"

    print(f"\n{'='*70}")
    print(f"  [{condition_name}] Co-retrieval 실험")
    print(f"{'='*70}")
    print(f"  link_k={link_k}, expand_links={expand_links}, top_k={top_k}")
    print(f"  Samples: {len(samples)}, Distractors: {len(distractors)}")
    print(f"  Gap sizes: {gap_sizes}")

    results = {}
    detailed_logs = []
    first_logged = False

    for gap in gap_sizes:
        print(f"\n  [Gap {gap}] 실험 중...")

        co_count = 0
        single_count = 0
        none_count = 0
        all_link_stats = []

        for sample_idx, sample in enumerate(tqdm(samples, desc=f"Gap={gap}", leave=False)):
            # 새 메모리 생성 (각 샘플마다 독립적)
            memory = LinkedMemory(encoder, link_k=link_k)

            # === 메모리 구축 (순차 저장) ===

            # 1. Prefix distractors (50개)
            n_prefix = min(50, len(distractors))
            prefix_indices = np.random.choice(len(distractors), n_prefix, replace=False)
            for i in prefix_indices:
                memory.store(distractors[i], create_links=(link_k > 0))

            # 2. Doc1 저장
            doc1_idx = memory.store(sample['doc1'], create_links=(link_k > 0))

            # 3. Gap distractors
            if gap > 0:
                n_gap = min(gap, len(distractors))
                gap_indices = np.random.choice(len(distractors), n_gap, replace=False)
                for i in gap_indices:
                    memory.store(distractors[i], create_links=(link_k > 0))

            # 4. Doc2 저장 (⭐ 이 시점에 Doc1과 링크가 생성될 수 있음!)
            doc2_idx = memory.store(sample['doc2'], create_links=(link_k > 0))

            # 5. Suffix distractors (20개)
            n_suffix = min(20, len(distractors))
            suffix_indices = np.random.choice(len(distractors), n_suffix, replace=False)
            for i in suffix_indices:
                memory.store(distractors[i], create_links=(link_k > 0))

            # === 검색 ===
            retrieved_idx, retrieved_scores = memory.retrieve(
                sample['question'],
                top_k=top_k,
                expand_links=expand_links
            )
            retrieved_set = set(retrieved_idx)

            # === 결과 평가 ===
            d1_found = doc1_idx in retrieved_set
            d2_found = doc2_idx in retrieved_set

            # Doc1-Doc2 링크 확인
            doc1_doc2_linked = (
                doc2_idx in memory.links.get(doc1_idx, []) or
                doc1_idx in memory.links.get(doc2_idx, [])
            )

            if d1_found and d2_found:
                co_count += 1
                result_type = "co-retrieval"
            elif d1_found or d2_found:
                single_count += 1
                result_type = "single"
            else:
                none_count += 1
                result_type = "none"

            # 링크 통계 수집
            all_link_stats.append(memory.get_link_stats())

            # 첫 번째 샘플 상세 로깅
            if verbose_first and not first_logged and sample_idx == 0:
                first_logged = True
                print(f"\n  {'─'*66}")
                print(f"  [상세 로그] 첫 번째 샘플 (Gap={gap}, {condition_name})")
                print(f"  {'─'*66}")

                print(f"\n  ▶ INPUT")
                print(f"    Question: {sample['question']}")
                print(f"    Doc1: {sample['doc1'][:100]}...")
                print(f"    Doc2: {sample['doc2'][:100]}...")

                print(f"\n  ▶ MEMORY 구성")
                print(f"    Total docs: {len(memory.embeddings)}")
                print(f"    Doc1 idx: {doc1_idx}, Doc2 idx: {doc2_idx}")
                print(f"    Doc1-Doc2 linked: {'✓' if doc1_doc2_linked else '✗'}")

                if link_k > 0:
                    print(f"\n  ▶ LINK 정보")
                    print(f"    Doc1의 링크: {memory.links.get(doc1_idx, [])}")
                    print(f"    Doc2의 링크: {memory.links.get(doc2_idx, [])}")
                    stats = memory.get_link_stats()
                    print(f"    총 링크 수: {stats.get('total_links', 0)}")
                    print(f"    문서당 평균 링크: {stats.get('avg_links_per_doc', 0):.2f}")

                print(f"\n  ▶ OUTPUT")
                print(f"    검색된 idx: {retrieved_idx[:10]}{'...' if len(retrieved_idx) > 10 else ''}")
                print(f"    검색된 수: {len(retrieved_idx)}")
                print(f"    Doc1 found: {'✓' if d1_found else '✗'}")
                print(f"    Doc2 found: {'✓' if d2_found else '✗'}")
                print(f"    Result: {result_type.upper()}")
                print(f"  {'─'*66}")

            # 상세 로그 저장
            if sample_idx < 3:
                detailed_logs.append({
                    'condition': condition_name,
                    'gap': int(gap),
                    'sample_idx': int(sample_idx),
                    'question': sample['question'],
                    'doc1_idx': int(doc1_idx),
                    'doc2_idx': int(doc2_idx),
                    'doc1_doc2_linked': bool(doc1_doc2_linked),
                    'retrieved_count': len(retrieved_idx),
                    'd1_found': bool(d1_found),
                    'd2_found': bool(d2_found),
                    'result': result_type,
                    'link_stats': memory.get_link_stats()
                })

        # Gap 결과 집계
        total = len(samples)

        # 평균 링크 통계
        avg_link_rate = np.mean([
            s.get('total_links', 0) / max(s.get('total_documents', 1), 1)
            for s in all_link_stats
        ]) if all_link_stats else 0

        results[gap] = {
            'co_retrieval_rate': co_count / total,
            'single_rate': single_count / total,
            'none_rate': none_count / total,
            'counts': {
                'co': co_count,
                'single': single_count,
                'none': none_count,
                'total': total
            },
            'avg_link_rate': avg_link_rate
        }

        print(f"      Co-retrieval: {co_count}/{total} ({100*co_count/total:.1f}%)")
        print(f"      Single:       {single_count}/{total} ({100*single_count/total:.1f}%)")
        print(f"      None:         {none_count}/{total} ({100*none_count/total:.1f}%)")

    return results, detailed_logs


def run_all_conditions(encoder, samples, distractors, gap_sizes, top_k=5):
    """모든 조건 실행 (Baseline + Link-1/3/5)"""

    all_results = {}
    all_logs = []

    conditions = [
        {'name': 'Baseline', 'link_k': 0, 'expand_links': False},
        {'name': 'Link-1', 'link_k': 1, 'expand_links': True},
        {'name': 'Link-3', 'link_k': 3, 'expand_links': True},
        {'name': 'Link-5', 'link_k': 5, 'expand_links': True},
    ]

    for cond in conditions:
        results, logs = run_linking_experiment(
            encoder, samples, distractors, gap_sizes,
            link_k=cond['link_k'],
            top_k=top_k,
            expand_links=cond['expand_links'],
            verbose_first=(cond['name'] == 'Link-3')  # Link-3에서만 상세 로깅
        )
        all_results[cond['name']] = results
        all_logs.extend(logs)

    return all_results, all_logs


def visualize_comparison(all_results, output_dir, config):
    """조건별 비교 시각화"""
    os.makedirs(output_dir, exist_ok=True)

    conditions = list(all_results.keys())
    gaps = sorted(all_results[conditions[0]].keys())

    # 색상 정의
    colors = {
        'Baseline': 'gray',
        'Link-1': 'blue',
        'Link-3': 'green',
        'Link-5': 'orange'
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Line plot: Co-retrieval rate by gap
    ax1 = axes[0]
    for cond in conditions:
        rates = [all_results[cond][g]['co_retrieval_rate'] * 100 for g in gaps]
        ax1.plot(gaps, rates, '-o', label=cond, color=colors.get(cond, 'black'),
                linewidth=2, markersize=8)

        # 각 점에 값 표시
        for x, y in zip(gaps, rates):
            ax1.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=9)

    encoder_name = config.get('encoder', 'sbert').upper()
    ax1.set_xlabel('Gap Size (# distractors between Doc1 and Doc2)', fontsize=11)
    ax1.set_ylabel('Co-retrieval Rate (%)', fontsize=11)
    ax1.set_title(f'Explicit Linking Effect on Co-retrieval ({encoder_name})', fontsize=13)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Bar plot: Gap별 조건 비교
    ax2 = axes[1]
    x = np.arange(len(gaps))
    width = 0.2

    for i, cond in enumerate(conditions):
        rates = [all_results[cond][g]['co_retrieval_rate'] * 100 for g in gaps]
        bars = ax2.bar(x + i * width, rates, width, label=cond,
                      color=colors.get(cond, 'black'), alpha=0.8)

        # 막대 위에 값 표시
        for bar, rate in zip(bars, rates):
            ax2.annotate(f'{rate:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Gap Size', fontsize=11)
    ax2.set_ylabel('Co-retrieval Rate (%)', fontsize=11)
    ax2.set_title('Comparison by Gap Size', fontsize=13)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([str(g) for g in gaps])
    ax2.legend()
    ax2.set_ylim([0, 105])

    plt.tight_layout()

    encoder_type = config.get('encoder', 'sbert')
    fig_path = os.path.join(output_dir, f'phase2_linking_{encoder_type}_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # JSON 저장
    json_path = os.path.join(output_dir, f'phase2_linking_{encoder_type}_results.json')
    if encoder_type == 'sbert':
        encoder_desc = 'sentence-transformers/all-MiniLM-L6-v2 (document-level cosine similarity)'
    else:
        encoder_desc = 'YuWangX/mplus-8b (token-level, sigmoid(q·k).mean(), Base Llama hidden states)'

    output_data = {
        'experiment': f'Phase 2 (B-3): Explicit Linking Effect ({encoder_type.upper()})',
        'timestamp': datetime.now().isoformat(),
        'method': {
            'description': 'Write-time Linking + Read-time Link Expansion',
            'encoder': encoder_desc,
            'retrieval': 'token-level sigmoid' if encoder_type == 'mplus' else 'document-level cosine',
            'conditions': ['Baseline', 'Link-1', 'Link-3', 'Link-5']
        },
        'config': config,
        'results': {
            cond: {str(g): v for g, v in results.items()}
            for cond, results in all_results.items()
        }
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n  [저장] {fig_path}")
    print(f"  [저장] {json_path}")

    return fig_path, json_path


def print_comparison_summary(all_results):
    """조건별 비교 요약 출력"""
    conditions = list(all_results.keys())
    gaps = sorted(all_results[conditions[0]].keys())

    print(f"\n")
    print(f"{'='*80}")
    print(f"  Phase 2 (B-3): Explicit Linking 실험 결과")
    print(f"{'='*80}")

    # 헤더
    header = f"  {'Gap':>6} |"
    for cond in conditions:
        header += f" {cond:>12} |"
    print(header)
    print(f"  {'-'*(8 + 15*len(conditions))}")

    # 각 Gap별 결과
    for gap in gaps:
        row = f"  {gap:>6} |"
        for cond in conditions:
            rate = all_results[cond][gap]['co_retrieval_rate'] * 100
            row += f" {rate:>11.1f}% |"
        print(row)

    print(f"  {'-'*(8 + 15*len(conditions))}")

    # 개선 효과 분석
    print(f"\n  [개선 효과 분석]")
    if 'Baseline' in conditions and 'Link-3' in conditions:
        for gap in gaps:
            baseline = all_results['Baseline'][gap]['co_retrieval_rate'] * 100
            link3 = all_results['Link-3'][gap]['co_retrieval_rate'] * 100
            improvement = link3 - baseline
            print(f"    Gap {gap}: Baseline {baseline:.1f}% → Link-3 {link3:.1f}% ({improvement:+.1f}%p)")

    print(f"\n{'='*80}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2: Explicit Linking Experiment")
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_distractors', type=int, default=1000)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./experiments/results_phase2_linking')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--single_condition', type=str, default=None,
                       help='Run single condition only (Baseline, Link-1, Link-3, Link-5)')
    parser.add_argument('--encoder', type=str, default='sbert', choices=['sbert', 'mplus'],
                       help='Encoder type: sbert (SentenceTransformer) or mplus (M+ retriever)')
    args = parser.parse_args()

    # Seed 설정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Quick 모드
    if args.quick:
        args.num_samples = 30
        args.num_distractors = 500
        gap_sizes = [0, 50, 200]
    else:
        gap_sizes = [0, 10, 50, 100, 200, 500]

    # Output dir에 encoder 타입 추가
    if args.encoder == 'mplus':
        args.output_dir = args.output_dir.replace('results_phase2_linking', 'results_phase2_linking_mplus')

    print(f"\n{'='*60}")
    print("Phase 2 (B-3): Explicit Linking Experiment")
    print(f"{'='*60}")
    print(f"Encoder: {args.encoder.upper()}")
    print(f"Samples: {args.num_samples}, Distractors: {args.num_distractors}")
    print(f"Top-k: {args.top_k}, Gaps: {gap_sizes}")

    # 데이터 로드
    samples, distractors = load_hotpotqa_data(args.num_samples, args.num_distractors)

    # 인코더 로드
    if args.encoder == 'sbert':
        print(f"\n[Encoder] SentenceTransformer 로드 중...")
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print(f"[Encoder] 로드 완료")
    else:
        print(f"\n[Encoder] M+ Retriever 로드 중...")
        encoder = MPlusEncoder()
        print(f"[Encoder] 로드 완료")

    # 실험 실행
    if args.single_condition:
        # 단일 조건만 실행
        cond_map = {
            'Baseline': (0, False),
            'Link-1': (1, True),
            'Link-3': (3, True),
            'Link-5': (5, True),
        }
        if args.single_condition not in cond_map:
            print(f"Unknown condition: {args.single_condition}")
            return

        link_k, expand = cond_map[args.single_condition]
        results, logs = run_linking_experiment(
            encoder, samples, distractors, gap_sizes,
            link_k=link_k, top_k=args.top_k, expand_links=expand
        )
        all_results = {args.single_condition: results}
    else:
        # 모든 조건 실행
        all_results, logs = run_all_conditions(
            encoder, samples, distractors, gap_sizes, top_k=args.top_k
        )

    # Config
    config = {
        'encoder': args.encoder,
        'num_samples': int(len(samples)),
        'num_distractors': int(len(distractors)),
        'top_k': int(args.top_k),
        'gap_sizes': [int(g) for g in gap_sizes],
        'seed': int(args.seed),
        'quick_mode': bool(args.quick),
    }

    # 결과 출력 및 저장
    print_comparison_summary(all_results)
    visualize_comparison(all_results, args.output_dir, config)

    print(f"\n  실험 완료! 결과: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
