# exp_003_diffrec — DiffRec (SIGIR 2023)

**Diffusion Recommender Model** — Wang, Xu, Feng, Lin, He, Chua. SIGIR 2023.

Week 1 Day 4-5. 4번째 paradigm — **generative (diffusion)**. CF (ALS/EASE) + Sequential (BSARec) 와 family-diverse 한 마지막 main 모델.

## 가설

| 항목 | 근거 |
|---|---|
| Diffusion family → ALS/EASE/BSARec 와 다른 시그널 잡음 | 생성 모델은 user의 "intent distribution" 자체를 학습. discriminative scoring 모델과 보완 |
| Ensemble v2 (4-paradigm RRF) 진정한 lift 후보 | ensemble_v1 negative result 의 정확한 반대: family-diverse 모델 결합 |
| RecBole 빌트인 → 구현 부담 작음 | BSARec 처럼 paper-to-code port 안 해도 됨 |

## 알고리즘 — Diffusion on Interaction Vector

```
forward (학습):
  x_0 = user u 의 binary interaction vector (size n_items = 29,413)
  sample t ~ uniform [1, T]
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon   # Gaussian noise
  loss = || x_0 - DNN(x_t, t) ||²                                    # predict clean x_0

reverse (추론):
  x_0 = user u 의 현재 interaction vector
  x_t = noise(x_0, t=T)            # add noise to existing history
  for t in T, T-1, ..., 1:
    x_{t-1} = reverse_step(x_t, t)
  predicted_x_0 = x_0_estimate     # 각 item 에 대한 score
  top-k items by predicted_x_0
```

**중요**: item embedding **없음**. interaction vector 자체 (29.5k dim) 가 입력/출력. DNN architecture: `[29.5k → dims_dnn (e.g., 300) → 29.5k] + timestep embedding`.

## 주요 hyperparameter ([config.yaml](config.yaml))

| 카테고리 | 항목 | 값 | 비고 |
|---|---|---|---|
| Diffusion | `steps` (T) | 5 | RecBole/논문 default. 작은 T로 충분 (recsys 데이터 특성) |
| | `noise_scale` | 0.001 | beta 스케일링 |
| | `noise_schedule` | linear | linear / cosine 선택 가능 |
| | `mean_type` | x0 | x_0 직접 예측 (vs noise epsilon) |
| Model | `dims_dnn` | [300] | 작은 MLP. 더 큰 hidden 시 lift 가능 |
| | `embedding_size` (timestep) | 10 | timestep embedding dim |
| | `mlp_act_func` | tanh | — |
| Train | batch | 2048 | exp_002 와 동일 |
| | learning_rate | 0.001 | DiffRec default (item embedding 없어서 LR 보수적) |

## 데이터 흐름

- **데이터 준비** (`data_prep.py`): exp_002 와 동일 패턴 — `train.parquet` time-split → `./data/cy_commerce/cy_commerce.inter`
- **학습**: RecBole `general_recommender` (sequential 아님). interaction matrix 단위로 학습
- **자체 val**: exp_000/001/002 와 동일 (`val_days=7`, `gt=['purchase']`, `eval_users=928`)
- **추론**: 638,257명 전원에 대해 user_id → DiffRec.full_sort_predict → top-50 → UUID 변환
- **Cold-start**: 학습 데이터에 안 보인 user 는 popularity fallback

## 메모리 / 시간 예상

- 모델 크기: `(29.5k × 300) × 2 + 작은 시간 embedding` ≈ **~17M params**. BSARec (2M) 보다 큼. item-wise dense 곱셈 때문
- GPU 메모리: 학습 시 batch=2048 × interaction vector 29.5k = ~250MB activation. 여유 충분
- 학습 시간: 한 epoch ~5분 추정 (BSARec 보다 약간 느림 — 더 큰 모델 + 5 diffusion step). 30 epoch 수렴 가정 시 **약 2.5-3시간**
- 추론: 638k user / batch 1024 ≈ ~10분

## 실행 (서버에서)

```bash
cd /root/workspace/recsys-commerce && git pull
cd experiments/exp_003_diffrec

# 1) 데이터 준비 (1분 이내)
python data_prep.py 2>&1 | tee data_prep.log

# 2) 학습 — nohup 으로 detach
nohup python train.py > train.log 2>&1 &
disown

# 3) 학습 끝나면 (예상 2.5-3시간)
nohup python inference.py > inference.log 2>&1 &
disown
```

## 결과 비교 (학습 후 작성)

| 모델 | NDCG@10 | recall@10 | 학습시간 | family |
|---|---:|---:|---:|---|
| exp_000 ALS | 0.1838 | 0.2558 | — | MF |
| exp_001 EASE | 0.1848 | 0.2909 | 5분 | item-item |
| exp_002 BSARec (4m) | TBD | TBD | TBD | sequential |
| exp_002b BSARec (4w) | TBD | TBD | TBD | sequential |
| **exp_003 DiffRec** | TBD | TBD | TBD | **diffusion** |

핵심 질문:
1. DiffRec NDCG가 단독으로 EASE/BSARec 와 어느 수준?
2. 4-paradigm RRF (ensemble_v2: ALS + EASE + BSARec + DiffRec) 가 ensemble_v1 (ALS+EASE) 의 negative result 를 뒤집는가?
3. T-DiffRec (`time-aware: true`) ablation 의 lift 가치?

## 다음 액션

- 결과 OK → `ensemble_v2_4paradigm/` 생성, 4모델 RRF + lift 측정
- DiffRec lift 작거나 시간 부담 시 LightGCN (Day 6) 으로 진입
- Week 1 마지막: ensemble 정리 + Week 2 service 준비

## 참고

- **논문**: Wang, W., Xu, Y., Feng, F., Lin, X., He, X., Chua, T.-S. (2023). "Diffusion Recommender Model." SIGIR 2023.
- **저자 코드**: https://github.com/YiyanXu/DiffRec (참고만, 우리는 RecBole 빌트인 사용)
- **RecBole 구현**: https://recbole.io/docs/recbole/recbole.model.general_recommender.diffrec.html
- [docs/candidate_models.md §5](../../docs/candidate_models.md)
