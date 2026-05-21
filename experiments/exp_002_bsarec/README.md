# exp_002_bsarec — BSARec (AAAI 2024)

**Beyond Self-Attention for Sequential Recommendation** — Shin, Choi, Wi, Park, AAAI 2024.

Week 1 Day 2-3. Sequential family 첫 모델 — ensemble_v1 negative result (ALS + EASE 둘 다 CF family라 다양성 부족)에 대한 직접적 해답.

## 가설

| 항목 | 근거 |
|---|---|
| BSARec sequential family → ALS/EASE 대비 잡는 정답 set이 다름 | user history의 시간 순서 시그널을 봄. CF 두 모델이 못 잡는 영역 보완 기대 |
| FRA (Frequency-Rescaling Attention)가 단순 SASRec 대비 lift | 99.78% view-dominated implicit feedback → noise 많은 sequence. FFT low-pass smoothing이 노이즈 완화에 효과적 |
| Ensemble RRF lift 재시도 가치 | CF (ALS/EASE) + Sequential (BSARec) 결합 = family-diverse → 진짜 RRF lift 가능성 |

## 알고리즘 — Hybrid 구현

```
┌─────────────────────────────────────────────┐
│ Data pipeline + Trainer: RecBole (재사용)   │
│ - SASRec_dataset/ (베이스라인이 이미 준비)   │
│ - 토크나이저 / data_preparation / Trainer   │
├─────────────────────────────────────────────┤
│ Model:                                       │
│   Item Embedding + Position Embedding       │
│        ↓                                     │
│   BSARecEncoder × n_layers:                  │
│     ─ FrequencyLayer (★ 직접 구현)          │
│     ─ MultiHeadAttention (RecBole 재사용)   │
│     ─ α * freq + (1-α) * attn               │
│     ─ FeedForward (RecBole 재사용)          │
│        ↓                                     │
│   Last position hidden state                │
│        ↓                                     │
│   Score = h @ Item Embedding^T (full sort)  │
└─────────────────────────────────────────────┘
```

**FRA 수식** ([fra.py](fra.py)):
```
X_freq    = rFFT(X, dim=L)
low_pass  = irFFT( zero_out_above_c(X_freq) )    # 저주파 성분만
high_pass = X - low_pass                          # 고주파 잔차
out       = LayerNorm( dropout(low_pass + (sqrt_beta²) * high_pass) + X )
```

`sqrt_beta`는 학습 가능 per-hidden-dim 파라미터, 제곱으로 비음수 보장.

`α` = frequency 성분 weight (논문 Beauty: 0.7).
`c` = low-pass cutoff frequency (논문 sweep: 1~9, 우리는 5에서 시작).

## 주요 hyperparameter ([config.yaml](config.yaml))

| 항목 | 값 | 비고 |
|---|---|---|
| max_seq_length | 50 | EDA p90=29, 50이 SASRec 표준 |
| n_layers | 2 | SASRec/BSARec 논문 모두 2 |
| n_heads | 2 | SASRec 표준 |
| hidden_size | 64 | SASRec 표준 |
| **alpha** | **0.7** | freq 비중. 노이즈 많은 sequence에 강함 |
| **c** | **5** | low-pass cutoff. 작을수록 강한 smoothing |
| loss_type | CE | softmax over full vocab — view-dominated에 적합 |
| batch | 256 | RTX 3090 hidden=64 / seq=50에 여유 |
| epochs | 200 | early stop step=10 |
| learning_rate | 0.001 | Adam default |

## 데이터 흐름

- **학습**: `/root/data/SASRec_dataset/` (베이스라인이 train.parquet → RecBole atomic 변환). user/item UUID가 RecBole 내부 int로 retokenize됨
- **자체 val** (exp_000/001 비교용): `core/validation.py time_based_split(val_days=7, gt=['purchase'])`로 우리 식 self-val. `eval_users` (928명) 재사용
- **추론**: 638,257명 전원에 대해 user history (UUID 공간) → RecBole int 매핑 → BSARec forward → top-50 → UUID 역매핑 → predictions.parquet
- **Cold-start**: 학습 데이터에 sequence 없는 user는 popularity fallback

## 메모리 / 시간 예상

- 모델 크기: hidden=64, n_layers=2 → 약 2-5M params (item embedding이 dominant: 29.5k × 64 = 1.9M)
- 학습: RTX 3090에서 epoch당 1-3분, early stop 50-80 epoch → 약 2-4시간
- 추론: 638k user / batch 1024 ≈ 623 batch × 0.5s = 약 5분

## 실행 (서버에서)

```bash
cd /root/workspace/recsys-commerce && git pull

cd experiments/exp_002_bsarec

# 학습 — wandb backup 포함
python train.py 2>&1 | tee train.log

# 추론 + 자체 val + submission CSV 생성
python inference.py 2>&1 | tee inference.log
```

## 산출물

- `saved/BSARec-<timestamp>.pth` — RecBole 체크포인트 (`.gitignored`)
- `saved/best_ckpt_path.txt` — inference가 찾을 체크포인트 경로
- `saved/wandb_run_id.txt` — train+inference 단일 run 통합
- `predictions.parquet` — top-50 + score (UUID 공간, ensemble 입력) `.gitignored`
- `output.csv` — 제출 파일 (638,257 × 10) `.gitignored`

## 결과 (학습 후 작성)

| 메트릭 | 값 | 비교 |
|---|---:|---|
| 자체 val NDCG@10 | TBD | exp_000 ALS 0.1838, exp_001 EASE 0.1848 |
| 자체 val recall@10 | TBD | ALS 0.2558, EASE 0.2909 |
| 학습 시간 | TBD | — |

## 다음 액션

- 결과 OK → `ensemble_v2_als_ease_bsarec/` 생성 (family-diverse 3-모델 RRF)
- α / c ablation 시간 여유 시 (논문 sweep): α ∈ {0.5, 0.7, 0.9}, c ∈ {3, 5, 7}
- Week 1 Day 4-5 → exp_003 DiffRec (paper-to-code)

## 참고 / Attribution

- **논문**: Shin, Y., Choi, J., Wi, H., Park, N. (2024). "An Attentive Inductive Bias for Sequential Recommendation Beyond the Self-Attention." AAAI 2024.
- **저자 코드**: https://github.com/yehjin-shin/BSARec (Apache-2.0). `fra.py`의 FrequencyLayer는 저자의 `src/model/bsarec.py`를 PyTorch 표준 LayerNorm 사용으로 minor 수정한 port
- **RecBole**: https://github.com/RUCAIBox/RecBole — `SequentialRecommender` 베이스, `MultiHeadAttention` + `FeedForward` 재사용
- [docs/candidate_models.md §5](../../docs/candidate_models.md) Week 1 plan
- [docs/eda_findings.md](../../docs/eda_findings.md) — sequence 특성 (CV=4.12, p90=29)
