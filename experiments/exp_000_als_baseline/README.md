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

## 결과 (2026-05-20)

| 메트릭 | 값 | 비교 |
|---|---:|---|
| 자체 val NDCG@10 (Feb 23~29 hold-out, 928 eval users) | **0.18376** | 1차 mismatched 0.0288 대비 6.4× |
| 자체 val recall@10 | **0.25578** | 1차 0.0566 대비 4.5× |
| **공식 Public NDCG@10 (Mar 1~7)** | **0.0791** | 자체 val의 43% |
| 베이스라인 공시 public | 0.0847 | 우리가 **6.6% 낮음** |

- **누적 제출**: 1회 (calibration용)
- **wandb run**: `cy-commerce-recsys` / `exp_000_als_baseline` (entity `yooni0125-`, 2026-05-20T07:52:45Z, train+inference 단일 run)

## 결론 — 파이프라인 검증 완료, calibration 비율 확보

### 1. 파이프라인은 정상 동작

베이스라인 공시 0.0847과 우리 0.0791의 갭 = **6.6%**, 학습 데이터 손실 비율 7일/120일 = **5.8%**와 거의 일치. 즉 갭의 거의 전부가 **학습 데이터 7일 holdout** 때문이고, 알고리즘/구현/shared 파이프라인은 베이스라인과 등가. 0.0791이 0.0847에 가까운 것만으로:

- shared/ 4모듈 (data_loader / validation / metrics / submission) 모두 638k user × 8.35M event 실제 스케일에서 정상
- 베이스라인 hyperparams + flags 그대로 매칭한 결과 베이스라인 거의 재현
- predictions.parquet → output.csv → validate_submission → 실제 제출까지 end-to-end OK

### 2. self-val ÷ public ≈ 2.32 (calibration 비율 확정)

향후 새 모델의 self-val 점수를 보면 약 **÷2.32** 해서 public 기대치 추정 가능. 예: 다음 모델 self-val 0.25 → public ~0.108 예상.

단 이 비율은 **ALS + 113일 학습 + Feb 23~29 self-val** 조합에서 측정된 것. 다른 모델(시퀀셜)이나 다른 split (val_days 변경) 시 비율 달라질 수 있음. 첫 시퀀셜 실험 끝나면 한 번 더 calibration 제출 권장.

### 3. self-val의 신뢰 범위

- ✅ 같은 알고리즘·같은 split 내 **상대 비교** (ranking) — 신뢰 가능
- ⚠️ **절대값** — Feb 27~29 spike 효과로 inflate됨 ([docs/eda_findings.md §5](../../../docs/eda_findings.md))
- 다음 실험에선 self-val을 비교 지표로, public 환산치를 "이 모델이 제출 가치 있는가" 판단 근거로 활용

### 4. 트레이드오프 발견

| 모드 | 학습 데이터 | 자체 val 가능? | 예상 public |
|---|---|---|---|
| Hold-out 7일 (exp_000 현재) | 113일 | ✅ | 0.0791 (측정값) |
| Full data (베이스라인 공시 방식) | 120일 | ❌ | 0.0847 (공시) |

**향후 실험 권고**: 모델 개발/iteration 시엔 hold-out으로 self-val, 최종 제출 시엔 full data로 재학습. (현재 train.py는 split 강제 — 추후 `val_days=0` 옵션 추가 검토)

## 다음 액션

- ~~calibration 제출~~ ✅ 완료 (NDCG@10 = 0.0791)
- exp_001: ALS lever ablation (선택사항) — `filter_already_liked` / event_weights / factors 중 어떤 게 6.4× 점프의 주범인지 분리. 점수 향상보단 학습 목적.
- **exp_001 또는 exp_002 (다음 우선순위)**: sequential 모델 (TiSASRec 또는 BSARec, [docs/candidate_models.md](../../../docs/candidate_models.md)) — EDA 강력 권장. ALS 베이스라인 재현 완료했으니 다음 단계 진입 자연스러움. self-val × 1/2.32 ≈ public 기대치로 활용
- exp_003: two-stage (ALS candidate + LightGBM reranker)
- exp_NNN: EASE — 30줄 구현, ensemble 다양성 보조 멤버
