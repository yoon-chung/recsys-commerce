# ensemble_v1_als_ease — RRF combine of ALS + EASE

Week 1 plan 의 첫 ensemble 체크포인트. 두 single-model 결과 (exp_000 ALS / exp_001 EASE) 를 **Reciprocal Rank Fusion** 으로 합쳐서, ensemble 파이프라인 검증 + 다양성 가설 (다른 family 모델 결합 시 lift) 확인.

## 가설

| 항목 | 근거 |
|---|---|
| RRF 가 두 모델 모두 대비 NDCG 향상 | exp_001 결과 — 동일 NDCG (0.1838 vs 0.1848) 인데 recall 은 EASE +13.7% → 두 모델이 잡는 정답 set 이 부분적으로 다름을 시사 |
| ALS (MF factor) ⊕ EASE (item-item) | 다른 family → 같은 item 에 대해 다른 rank 부여할 가능성 큼 |
| RRF k_const=60 (paper default) 충분 | top-50 후보만 사용하므로 k=60 이면 rank 1과 rank 50 의 기여 차이 약 50배. ablation 여지는 있음 |

## 입력 / 출력

**입력** (Week 1 single-model 결과 재활용):
- `../exp_000_als_baseline/predictions.parquet` — 638,257 × top-50
- `../exp_001_ease/predictions.parquet` — 623,866 × top-50 (cold-start 14,391 은 inference 단에서 popularity 로 채워졌으나, predictions.parquet 에는 알려진 user 만 들어있음)
- `../exp_001_ease/saved/val_gt.parquet`, `eval_users.json`, `mappings/` — 동일 split 재활용

**출력**:
- `fused_predictions.parquet` — RRF top-50 (`.gitignored`)
- `output.csv` — 제출용 (popularity fallback 적용, `.gitignored`)

## 알고리즘

```
for each user u:
    candidates = union of items from each model's top-50 for u
    for each candidate i:
        RRF(u, i) = sum over models m of  1 / (k_const + rank_m(u, i))
    sort candidates by RRF desc, keep top_n
```

- 한 모델만 추천한 item 도 정상 처리 (다른 모델 기여 0)
- rank 기반이라 모델 점수 scale calibration 불필요 (heterogeneous 모델에 안전)

## 실행 (서버에서)

```bash
cd /root/workspace/recsys-commerce && git pull

cd experiments/ensemble_v1_als_ease
python run.py
```

로컬에서 hyperparameter 만 바꿔 실험 시:

```bash
python run.py --k-const 30 --no-submission   # k_const ablation, 점수만 확인
python run.py --k-const 100 --no-submission
```

## 메트릭 (2026-05-21 실행 완료)

| 모델 | NDCG@10 | recall@10 |
|---|---:|---:|
| exp_000 ALS | 0.18376 | 0.25578 |
| exp_001 EASE | 0.18476 | 0.29089 |
| **RRF (k=60)** | **0.17253** | **0.26243** |

**Lift**:
- fused NDCG vs ALS: **−0.01123** (−6.1%)
- fused NDCG vs EASE: **−0.01223** (−6.6%)
- fused recall vs EASE: **−0.02847** (−9.8%)
- fused recall vs ALS: +0.00665 (+2.6%) — ALS recall 만 살짝 보충

→ **둘 다보다 더 나빠진 negative result**. 판정 기준 중 "fused NDCG < min" 케이스에 해당.

## 해석 — 왜 떨어졌나

**근본 원인은 다양성 부족 + 한 모델 dominance**:

1. **EASE 가 ALS 를 거의 dominate** — NDCG 비등(+0.5%) 인데 recall 은 EASE 가 **+13.7%**. EASE 가 잡는 정답 중 ALS 가 못 잡는 게 많고 반대는 적다는 뜻. 약한 ALS 시그널을 동등 가중치로 섞으면 noise 추가.

2. **두 모델 같은 family** — ALS (MF factor) 와 EASE (item-item) 가 다른 알고리즘이지만 **둘 다 implicit-feedback collaborative**. 같은 view/cart/purchase 데이터로 비슷한 시그널을 약간 다른 각도로 봄. 가설은 "다른 family → 다양성" 이었지만, 실제 시그널 다양성은 우리가 기대한 것보다 작음.

3. **k_const=60 이 top-50 에선 평탄** — 1/(60+1) vs 1/(60+50) = 1/61 vs 1/110, ratio 1.8 배. rank 신호 약해서 두 모델 합의 (both ranked high) 효과보다 individual top-rank 정답이 평균화로 밀려나는 손실이 큼.

4. **NDCG vs recall 분리 관찰** — recall 은 ALS 대비 살짝 오름 (+2.6%) 인데 NDCG 는 큰 폭 하락. 의미: RRF 가 더 다양한 정답 set 을 잡긴 하지만, 그 정답들을 top 으로 끌어올리지 못함. → fusion 으로 후보 set 다양화 효과는 약하게나마 있음.

## 학습 가치

negative result 지만 Week 1 plan 검증 측면에서 **유의미한 결과**:
- **다음 모델은 family-diverse 가 핵심** — Week 1 plan 의 BSARec (sequential, user-history 시간), DiffRec (generative), LightGCN (graph) 가 단순 더 많은 CF 대비 ensemble lift 가능성 큼
- 5 개 모델 모두 합친 ensemble 에서 **per-model weight + k_const tuning 필요**할 가능성 — 단순 RRF 가 항상 win 하지 않음을 데이터로 확인
- 포트폴리오 스토리: *"단순 RRF 가 작동 안 한 negative case 를 발견 → family diversity 가설 검증 후 sequential/content/graph 모델로 다양성 확보"*

## 다음 액션

- ✅ ALS+EASE RRF 실험 완료 (단순 fusion 한계 확인)
- ❌ submission 제출 안 함 (둘 다보다 나쁨)
- **Week 1 Day 2-3 → exp_002 BSARec** (sequential family, ensemble 다양성 핵심)
- exp_002 끝나면 `ensemble_v2_als_ease_bsarec/` 으로 확장. k_const ablation + weighted RRF 도 그때 같이 시도
- 후속 모델 추가될 때마다 `ensemble_vN_<list>/` 폴더로 누적

## 참고

- RRF 논문 + 모델 출처 (ALS / EASE): [docs/references.md §1 + ensemble methods](../../docs/references.md#1-추천-모델--논문--공식-코드)
- [docs/candidate_models.md §5](../../docs/candidate_models.md) — Week 1 5-모델 plan
- 입력 실험: [exp_000_als_baseline/README.md](../exp_000_als_baseline/README.md), [exp_001_ease/README.md](../exp_001_ease/README.md)
