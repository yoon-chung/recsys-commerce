# exp_000_als_baseline

implicit ALS **베이스라인 정확 재현** — 점수 향상이 아니라 (1) reference 점수 확보, (2) `shared/` 인프라 검증, (3) 자체 val ↔ public NDCG@10의 gap 측정이 목적.

> **1차 시도 노트 (2026-05-20)**: 처음에 factors=128 / reg=0.01 / alpha=40 / weights 1·3·5 / `filter_already_liked=True`로 돌려 self-val NDCG=0.0288, recall=0.0566 (베이스라인 공시 0.0847의 1/3 수준). hyperparams가 베이스라인과 어긋난 게 원인일 가능성이 가장 컸음. 그래서 이 폴더의 config을 베이스라인과 동일한 셋으로 갈아끼우고 재실행. 1차 시도의 wandb run은 entity `yooni0125-` / project `cy-commerce-recsys` / name `exp_000_als_baseline`(이전 run id)에 보존됨.

## 가설 (목적)

1. **베이스라인 ALS의 자체 val NDCG@10 측정**: 베이스라인 코드 (`/root/code/train_als.py`)와 hyperparams/플래그 동일하게 맞춰 돌리면, 같은 알고리즘이 우리 self-val(마지막 7일 hold-out)에서 어떤 수치를 내는가.
2. **gap 정량화**: 베이스라인 공시 public 0.0847과 우리 self-val 수치의 차이가 (a) 분포 shift (Feb 27~29 purchase 스파이크) 때문인지 (b) 우리 파이프라인 버그 때문인지 판단.
3. **`shared/` 모듈 통합 검증**: data_loader / validation / metrics / submission 4개가 638k user × 8.35M event 실제 스케일에서 동작 (1차 시도에서 이미 확인됨, 이번에도 재확인).

## 접근

- 모델: `implicit.als.AlternatingLeastSquares` (GPU).
- Train 입력: train_df (cutoff = `event_time.max() - 7 days` 이전) 의 view/cart/purchase 이벤트.
  - 베이스라인과 동일하게 모든 row label=1, (user,item) 합산 → 자연스럽게 "상호작용 횟수"가 confidence가 됨. EDA상 view가 99.78%라 view 다중방문이 자연 가중됨.
- ID 매핑: train_df에 등장한 user/item만 학습 universe (623,866 user / 29,413 item 측정됨). val window-only user (~14k)는 submission 시점에 popularity_fallback 채움.
- Eval: train_df 외 마지막 7일의 purchase 이벤트 = ground truth (1,443 rows / 1,105 users / 928 eval after restrict_to_train). NDCG@10 binary relevance.
- 베이스라인 코드는 **참조 금지** — implicit 라이브러리 컨벤션 따라 직접 구현.

## 주요 하이퍼 ([config.yaml](config.yaml))

| 항목 | 값 | 베이스라인과 비교 |
|---|---|---|
| factors | **32** | 동일 |
| regularization | **0.001** | 동일 |
| iterations | 15 | 동일 (implicit 기본값) |
| alpha | **10** | 동일 |
| event_weights (view/cart/purchase) | **1 / 1 / 1** | 동일 (베이스라인 label=1 groupby sum) |
| filter_already_liked_items | **false** | 동일 (key flag — 이미 본 item도 추천 가능) |
| use_gpu | true | 베이스라인 False, GPU만 다름 (알고리즘 동일) |
| val_days | 7 | 베이스라인에는 self-val 없음. 우리만 측정 |
| top_n | 50 | predictions.parquet candidate 수 |
| seed | 42 | 동일 |

## 실행 (서버에서)

```bash
cd /root/workspace/recsys-commerce && git pull

cd members/cy/exp_000_als_baseline
rm -rf saved wandb predictions.parquet output.csv   # 1차 시도 상태 정리
python train.py
python inference.py
```

`shared/` 모듈은 `sys.path` 자동 추가됨 (`Path(__file__).resolve().parents[3]` 트릭). wandb는 `wandb_run_id.txt`로 train+inference 단일 run 통합됨.

## 산출물

- `saved/als.npz`, `saved/interactions.npz`, `saved/mappings/`, `saved/val_gt.parquet`, `saved/eval_users.json`, `saved/wandb_run_id.txt` (.gitignored)
- `predictions.parquet` — top-50 + score (.gitignored, wandb backup)
- `output.csv` — 제출 파일, shape (6,382,570, 2) (.gitignored, 로컬 백업)

## 결과 (실행 후 채워넣기)

- **자체 validation NDCG@10**: `_._____` (last 7 days hold-out, restrict_to_train=True, gt=purchase)
- **자체 validation recall@10**: `_._____`
- **공식 Public NDCG@10**: 제출 시 (베이스라인 공시 = 0.0847)
- **wandb run**: `cy-commerce-recsys` / `exp_000_als_baseline` (entity `yooni0125-`)

## 결론 / 다음 액션 (실행 후)

다음 분기 판단 기준:

- **self-val ≈ 0.0847에 근접**: 인프라/파이프라인 모두 OK + 1차 시도의 hyperparams가 갭의 진짜 원인 → 다음 실험은 더 강한 모델 또는 ensemble로 진입
- **self-val ≪ 0.0847 (예: 0.03~0.05)**: 분포 shift가 메인. Feb 27~29 spike가 self-val을 어렵게 만듦 → public 점수만 신뢰. 추후 실험은 public 기준으로 평가 우선
- **self-val ≪ 0.0288 (재시도가 더 나빠짐)**: hyperparam 매치를 잘못 했거나 파이프라인 어딘가 버그 → diff 다시 검토

이후 후보:
- exp_001: ALS lever ablation (필요시) — factors / event_weights / filter_already_liked 한 번에 하나씩
- exp_002: BSARec / TiSASRec 등 sequential (EDA 강력 권장)
- exp_003: two-stage (ALS candidate + LightGBM reranker)
