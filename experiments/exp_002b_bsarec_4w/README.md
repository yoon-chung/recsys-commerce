# exp_002b_bsarec_4w — BSARec recency-window ablation

[exp_002_bsarec](../exp_002_bsarec/) 와 **모든 모델/하이퍼파라미터 동일**, 단 **학습 데이터를 마지막 4주 (Feb 1-29)** 로 제한한 ablation.

멘토 권고 ("구매가 몰린 짧은 기간으로 축소") 검증. 우리 데이터 특성 + Week 1 시간 제약을 고려해 3일이 아닌 **4주** 로 시작 (cold-start 폭증 vs recency 효과 균형).

## 가설

**Primary**: 4주 window 학습 BSARec 는 활성 user (sequence ≥ 1 in 4-week window) 부분집합에서 4개월 baseline 대비 NDCG@10 lift.

**Secondary**: 전체 638k user 기준 NDCG 는 변동 작거나 살짝 down — 50-70% user 가 cold-start 로 빠지면서 popularity fallback 비중이 커짐.

**근거**:
- BSARec sequence는 이미 `max_seq_length=50` 으로 cap 됐지만, **item embedding 은 모든 interaction 으로 학습**됨. 4주 데이터 → item embedding 이 Mar 1-7 예측 타깃 분포에 더 align
- EDA Feb 27-29 spike: 마지막 3일에 구매 집중. 4주 window 가 그 spike + 직전 reference 기간 모두 포함
- recency 강한 sequential 모델 family 특성

## 데이터 비교

| 항목 | exp_002 (4-month) | exp_002b (4-week) |
|---|---:|---:|
| Train 기간 | Nov 1 – Feb 29 (~120일) | Feb 1 – Feb 29 (28일) |
| Train rows | 8M | ~2M (추정 25%) |
| 활성 user (≥1 event) | 623,866 | 200k-300k (추정) |
| Cold-start user (popularity 대상) | 14,391 | 340k-440k (추정) |
| Item vocab 크기 | 29,413 | 25k-28k (추정, 일부 희소 item 누락) |

## 핵심 변경점

[exp_002_bsarec](../exp_002_bsarec/) 와 diff:
- **data_prep.py**: default `--last-days 28`, output `./data/cy_commerce_4w/`
- **config.yaml**: `dataset: cy_commerce_4w`, `run_name: exp_002b_bsarec_4w`
- **그 외 (train.py / inference.py / fra.py / bsarec_model.py)**: **완전 동일** — fair ablation

## 실행 (서버에서)

```bash
cd /root/workspace/recsys-commerce && git pull
cd experiments/exp_002b_bsarec_4w

# 1) 4-week 데이터셋 빌드 (1분 이내)
python data_prep.py 2>&1 | tee data_prep.log

# 2) 학습 — exp_002 (4-month) 와 같은 하이퍼파라미터, 데이터만 다름
nohup python train.py > train.log 2>&1 &
disown

# 3) 학습 끝나면 (예상 ~1시간, 데이터 작아서 더 빠를 듯)
nohup python inference.py > inference.log 2>&1 &
disown
```

## 결과 비교표 (실행 후 작성)

| 모델 / 메트릭 | NDCG@10 | recall@10 | 학습 시간 |
|---|---:|---:|---:|
| exp_000 ALS | 0.1838 | 0.2558 | — |
| exp_001 EASE | 0.1848 | 0.2909 | ~5분 |
| exp_002 BSARec (4-month) | TBD | TBD | TBD |
| **exp_002b BSARec (4-week)** | TBD | TBD | TBD |

핵심 질문:
1. 4w NDCG가 4m 대비 **lift**? → recency 가설 검증
2. 4w recall 차이가 NDCG 차이와 같은 방향? → 후보 set vs 순위 효과 분리
3. 활성 user 부분집합 (4w 데이터에 있는 user) 한정 ablation 도 가능 → 이건 inference.py 출력에 user별 메트릭 추가 필요 (지금은 안 함, 결과 나오면 결정)

## 다음 액션

- 결과 OK + lift 보이면 → `--last-days 14`, `--last-days 7` 추가 ablation
- 4w lift 없거나 4m 이 더 좋으면 → "**recency window는 helping 하지 않음**" portfolio 인사이트, 더 narrow한 window 시도 무의미
- 어느 쪽이든 → ensemble_v2 시 BSARec 4-month 가 더 fair한 ensemble 입력일 가능성 ↑

## 참고

- 부모 실험: [exp_002_bsarec](../exp_002_bsarec/README.md)
- [docs/eda_findings.md](../../docs/eda_findings.md) — Feb 27-29 spike, sequence 길이 분포
- 멘토 권고: 2026-05-21 멘토링 — "구매 몰린 짧은 기간으로 축소 학습"
