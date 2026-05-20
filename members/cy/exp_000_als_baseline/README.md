# exp_000_als_baseline

implicit ALS 베이스라인 재현 — **점수 향상 목적이 아닌 인프라 검증 + reference 점수 확보**.

## 가설 (목적)

1. **자체 val NDCG@10 reference 측정**: train의 마지막 7일(2020-02-23 ~ 2020-02-29) hold-out에서 ALS가 얼마나 나오는지. 이후 모든 실험이 이 점수보다 의미있게 좋아야 제출 가치 있음.
2. **`shared/` 모듈 통합 검증**: data_loader / validation / metrics / submission 4개 모듈이 638k user × 8.35M event 실제 스케일에서 정상 동작하는지 확인.
3. **end-to-end 파이프라인 점검**: model → predictions.parquet → output.csv → validate_submission 흐름이 막힘없이 도는지.

## 접근

- 모델: `implicit.als.AlternatingLeastSquares` (GPU)
- Train 입력: train_df (cutoff = `train.event_time.max() - 7 days` 이전)의 view/cart/purchase 이벤트를 confidence-weighted CSR로 변환
  - weights: `view=1`, `cart=3`, `purchase=5` (같은 (user,item) 다중 이벤트는 합산)
- ID 매핑: train_df에 등장한 user/item만 학습 universe로 (~608k user / 30k item 추정). 나머지 ~30k cold-start user는 submission 시점에 popularity_fallback 채움.
- Eval: train_df 외 마지막 7일의 purchase 이벤트 = ground truth. NDCG@10 binary relevance.
- 베이스라인 코드 (`/root/code/train_als.py`)는 **참조 금지** — implicit 라이브러리 컨벤션 따라 직접 구현.

## 주요 하이퍼 ([config.yaml](config.yaml))

| 항목 | 값 | 비고 |
|---|---|---|
| factors | 128 | matrix factorization rank |
| regularization | 0.01 | L2 |
| iterations | 15 | ALS 수렴 |
| alpha | 40 | confidence scaling (Hu et al. 2008) |
| use_gpu | true | RTX 3090 |
| val_days | 7 | 평가 윈도우 = 1주 |
| top_n | 50 | predictions.parquet 후보 수 |
| seed | 42 | reproducibility |

## 실행 (서버에서)

```bash
cd /root/workspace/recsys-commerce
git pull

cd members/cy/exp_000_als_baseline
python train.py           # ./saved/에 model + interactions + val_gt + eval_users 저장
python inference.py       # ./predictions.parquet + ./output.csv + 자체 val NDCG@10
```

`shared/` 모듈은 `sys.path` 자동 추가됨 (`Path(__file__).resolve().parents[3]` 트릭).

## 산출물

- `saved/als.npz` — 학습된 ALS 모델 (.gitignored)
- `saved/interactions.npz` — train CSR matrix (.gitignored)
- `saved/mappings/{user2idx,item2idx}.json` — ID 매핑 (.gitignored)
- `saved/val_gt.parquet` — hold-out purchase events (.gitignored)
- `saved/eval_users.json` — eval 대상 user 리스트 (.gitignored)
- `predictions.parquet` — top-50 candidates per known user (.gitignored, wandb backup)
- `output.csv` — 제출 파일, shape (6382570, 2) (.gitignored, 로컬 백업)

## 결과 (실행 후 채워넣기)

- **자체 validation NDCG@10**: `_._____` (last 7 days hold-out, restrict_to_train=True, gt=purchase)
- **자체 validation recall@10**: `_._____`
- **공식 Public NDCG@10**: 제출 시 채워넣기 (베이스라인 공시 = 0.0847)
- **wandb run**: `cy-commerce-recsys` / `exp_000_als_baseline`
  - artifacts: 모델 + predictions

## 결론 / 다음 액션 (실행 후)

- (자체 val이 0.0847에 가까운지 확인 → 가깝다면 split + 평가 파이프라인 신뢰할 만함)
- (자체 val NDCG와 public NDCG의 gap을 측정 → 이후 실험의 "유의미한 개선" 기준 정립)
- (다음 실험은: ... — exp_001부터)
