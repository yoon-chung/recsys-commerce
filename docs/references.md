# References

본 프로젝트에서 사용한 모든 외부 자료 (논문 / 저자 코드 / 프레임워크 / API 문서) 의 단일 출처. 새 실험을 만들 때는 여기에 항목 먼저 추가하고, 각 `experiments/exp_NNN_*/README.md` 의 "참고" 섹션에서 이 문서 항목으로 링크.

---

## 1. 추천 모델 — 논문 & 공식 코드

| 모델 | 논문 (저자/연도/Venue) | arXiv | 공식 코드 | License | RecBole 빌트인 | 우리 실험 |
|---|---|---|---|---|---|---|
| **ALS** | Hu, Koren, Volinsky (2008). "Collaborative Filtering for Implicit Feedback Datasets." ICDM | — | [benfred/implicit](https://github.com/benfred/implicit) | MIT | implicit lib | [exp_000_als_baseline](../experiments/exp_000_als_baseline/) |
| **EASE** | Steck, H. (2019). "Embarrassingly Shallow Autoencoders for Sparse Data." WWW | [1905.03375](https://arxiv.org/abs/1905.03375) | (논문 §3 closed-form 30 줄) | — | ✗ (직접 구현) | [exp_001_ease](../experiments/exp_001_ease/) |
| **BSARec** | Shin, Choi, Wi, Park (2024). "An Attentive Inductive Bias for Sequential Recommendation Beyond the Self-Attention." AAAI | [2312.10325](https://arxiv.org/abs/2312.10325) | [yehjin-shin/BSARec](https://github.com/yehjin-shin/BSARec) | Apache-2.0 | ✗ (FRA 직접 port) | [exp_002_bsarec](../experiments/exp_002_bsarec/), [exp_002b_bsarec_4w](../experiments/exp_002b_bsarec_4w/) |
| **DiffRec** | Wang, Xu, Feng, Lin, He, Chua (2023). "Diffusion Recommender Model." SIGIR | [2304.04971](https://arxiv.org/abs/2304.04971) | [YiyanXu/DiffRec](https://github.com/YiyanXu/DiffRec) | (저자 repo 확인) | ✓ `general_recommender.DiffRec` | [exp_003_diffrec](../experiments/exp_003_diffrec/) |
| **LightGCN** (planned) | He, Deng, Wang, Li, Zhang, Wang (2020). "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR | [2002.02126](https://arxiv.org/abs/2002.02126) | [kuandeng/LightGCN](https://github.com/kuandeng/LightGCN) | — | ✓ `general_recommender.LightGCN` | (Week 1 Day 6 예정) |

**ensemble 방법**:
- **RRF (Reciprocal Rank Fusion)**: Cormack, Clarke, Büttcher (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods." SIGIR. — [ensemble_v1_als_ease/](../experiments/ensemble_v1_als_ease/)

---

## 2. 프레임워크 / 라이브러리

| 도구 | 용도 | 공식 사이트 / docs |
|---|---|---|
| **RecBole** | 추천 모델 학습/평가 통합 프레임워크 (BSARec backbone, DiffRec 빌트인) | [recbole.io](https://recbole.io) / [GitHub](https://github.com/RUCAIBox/RecBole) |
| RecBole SASRec docs | [recbole.model.sequential_recommender.SASRec](https://recbole.io/docs/recbole/recbole.model.sequential_recommender.sasrec.html) — exp_002 backbone | |
| RecBole DiffRec docs | [recbole.model.general_recommender.DiffRec](https://recbole.io/docs/recbole/recbole.model.general_recommender.diffrec.html) — exp_003 직접 사용 | |
| RecBole MultiHeadAttention/FeedForward | [recbole.model.layers](https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py) — exp_002 BSARecLayer 에서 재사용 | |
| **implicit** | ALS 빠른 구현 (Cython, multi-thread) | [benfred/implicit](https://github.com/benfred/implicit) |
| **PyTorch** | 모델 backbone (BSARec, DiffRec 모두) | [pytorch.org](https://pytorch.org) |
| **wandb** | 학습 추적 + artifact 백업 (대회 종료 시 서버 회수 대비) | [wandb.ai](https://wandb.ai) project `cy-commerce-recsys` |
| **pandas / pyarrow** | parquet I/O (`/root/data/train.parquet` 8.35M rows) | — |
| **scipy.sparse** | user-item CSR matrix (EASE/ALS) | — |
| **scipy.linalg** | `cho_factor` + `cho_solve` — EASE 폐형 해 (positive definite 행렬 역행렬) | — |

### 핵심 API 시그니처 메모

**RecBole `Config(model=<class>, ...)`** — 문자열 대신 모델 클래스 직접 전달 트릭. `get_model()` 호출 + `exlib_recommender` (lightgbm) import 회피. [exp_002/train.py L73](../experiments/exp_002_bsarec/train.py), [exp_003/train.py L60](../experiments/exp_003_diffrec/train.py) 에서 사용.

**RecBole atomic file 포맷** — `<dataset_name>.inter` TSV, 컬럼명 `<field>:<dtype>` 형식 (e.g., `user_id:token`, `timestamp:float`). `data_prep.py` 가 모두 이 포맷 출력.

**RecBole `dataset.field2token_id / field2id_token`** — 사용자/아이템 UUID ↔ RecBole 내부 int ID 변환 dict. exp_002/exp_003 inference 에서 우리 UUID 공간 ↔ RecBole 공간 bridge 용.

---

## 3. 대회 / 데이터셋

| 항목 | 내용 |
|---|---|
| 대회 | 패스트캠퍼스 Upstage AI Lab RecSys 경진대회 |
| 태스크 | 4개월 이커머스 행동 로그 → 다음 1주일 구매 아이템 top-10 예측 |
| 메트릭 | NDCG@10 (binary relevance), public/private 50:50 random split |
| 데이터 규모 | 8,350,311 rows × 8 cols, 638,257 users × 29,502 items |
| 베이스라인 점수 | ALS = 0.0847 / SASRec = 0.0842 |
| 동점 규정 | 동점 시 제출 횟수 적은 쪽 우위 → 무의미한 제출 회피 |

**데이터 스키마**: `[CLAUDE.md](../CLAUDE.md)` 참조.

---

## 4. 멘토링 / 의사결정 메모

### 2026-05-21 멘토링 — 핵심 결정

- **방향성 전환**: 대회 점수 짜내기 < 추천 시스템 대표 모델 학습 + Week 2 서비스 foundation + 포트폴리오. ML 영역 (LightGBM/XGBoost ranker, 무거운 FE) 은 제외
- **Week 분할**: Week 1 = 모델링 (EASE / BSARec / DiffRec / LightGCN + ALS 완료) / Week 2 = FastAPI 서비스
- **기간 축소 ablation 권고**: "구매 몰린 짧은 기간으로 축소 학습" → [exp_002b_bsarec_4w](../experiments/exp_002b_bsarec_4w/) 에서 4주 ablation 으로 구체화
- **데이터 증강** (보류): Solar API 활용 방향 — item content embedding / conversational 서비스 layer 등은 추후 결정

### 의사결정 기록

- **앙상블 family 다양성 가설** ([ensemble_v1 negative result](../experiments/ensemble_v1_als_ease/README.md)): ALS + EASE 만의 RRF 는 same-family dominance 로 fused NDCG 가 둘 다보다 낮음 (-0.011). 4-paradigm (CF + sequential + diffusion + graph) 결합 시 lift 기대 → Week 1 plan 의 핵심 근거

---

## 5. 외부 자료 (도움된 search/fetch URLs)

학습 중 의사결정에 사용한 자료:
- [recbole-DiffRec.yaml (config defaults)](https://github.com/RUCAIBox/RecBole/blob/master/recbole/properties/model/DiffRec.yaml) — exp_003 하이퍼파라미터 default 확인
- [BSARec author src/model/bsarec.py](https://github.com/yehjin-shin/BSARec/blob/main/src/model/bsarec.py) — FrequencyLayer 정확한 수식
- [DiffRec paper PDF (Wang 2023)](https://hexiangnan.github.io/papers/sigir23-DiffRec.pdf) — 알고리즘 설명

---

## 어떻게 이 문서 업데이트하나

- 새 모델 실험을 시작할 때, 먼저 §1 표에 row 추가
- 새 도구/API를 도입하면 §2 에 추가
- 의사결정 (e.g., 멘토링, 가설 수정) 은 §4 에 날짜 + 한 줄 요약
- 각 실험 README 의 "참고" 섹션은 **이 문서 항목으로 링크** (중복 작성 X)
