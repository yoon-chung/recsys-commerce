#!/bin/bash
# exp_010b_lgbm_lambdarank / run_all.sh -- train_cv + inference 순차 실행
#
# 사용 (nohup):
#   cd experiments/exp_010b_lgbm_lambdarank
#   nohup bash run_all.sh > /root/exp_010b_run.log 2>&1 &
#   disown
#
# Monitor:
#   tail -f /root/exp_010b_run.log

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "=================================="
echo "exp_010b LambdaRank pipeline"
echo "Start: $(date)"
echo "=================================="

echo ""
echo "[1/2] train_cv.py (LambdaRank 5-fold) ..."
python train_cv.py 2>&1 | tee train.log

echo ""
echo "[2/2] inference.py (638k users) ..."
python inference.py 2>&1 | tee inference.log

echo ""
echo "=================================="
echo "DONE: $(date)"
echo "=================================="
echo ""
echo "Results:"
echo "  - CV summary:    saved/cv_results.json"
echo "  - Predictions:   predictions.parquet"
echo "  - Submission:    output.csv"