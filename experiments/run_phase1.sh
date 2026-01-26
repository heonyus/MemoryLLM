#!/bin/bash
# Phase 1: M+ Retriever Co-retrieval 분석 실행 스크립트
# GCP L4 GPU 환경

echo "=================================================="
echo "Phase 1: M+ Retriever Co-retrieval Analysis"
echo "=================================================="

# 1. 환경 설정
echo ""
echo "[Step 1] 환경 설정..."
pip install -q -r requirements_phase1.txt

# 2. 실행 옵션 선택
# Option A: M+ Retriever 사용 (권장, weights 추출 필요)
# Option B: SentenceBERT fallback 사용 (빠름, M+ weights 없어도 됨)

echo ""
echo "실행 옵션:"
echo "  A) M+ Retriever (정확, 느림) - weights 추출 필요"
echo "  B) SentenceBERT (빠름) - fallback"
echo ""

# Quick test with SentenceBERT (추천: 먼저 이걸로 테스트)
echo "[Step 2] Quick test (SentenceBERT)..."
python phase1_coretrieval_experiment.py \
    --use_sentence_bert \
    --quick_test \
    --output_dir ./results_phase1_quick

echo ""
echo "Quick test 완료!"
echo "결과: ./results_phase1_quick/"
echo ""

# Full experiment (선택)
read -p "전체 실험 실행? (y/n): " run_full
if [ "$run_full" = "y" ]; then
    echo ""
    echo "[Step 3] 전체 실험 실행..."
    python phase1_coretrieval_experiment.py \
        --use_sentence_bert \
        --num_samples 200 \
        --num_distractors 2000 \
        --output_dir ./results_phase1
    
    echo ""
    echo "전체 실험 완료!"
    echo "결과: ./results_phase1/"
fi

echo ""
echo "=================================================="
echo "완료!"
echo "=================================================="
