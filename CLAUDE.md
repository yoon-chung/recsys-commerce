# Commerce Behavior Purchase Prediction

부트캠프 RecSys 경진대회 작업용 프로젝트. 이커머스 4개월 행동 로그(view/cart/purchase)로 **다음 1주일에 사용자가 구매할 아이템 10개**를 예측한다.

**현재 작업 형태**: cy 개인 작업. 추후 팀 합류 가능성에 대비해 `members/cy/` 구조 채택 (팀 repo 이전 비용 최소화).

**Repo 상태**: 현재 **private** (GitHub: `yoon-chung/recsys-commerce`). 향후 대회 종료 후 **public 전환 예정**. ⚠️ 주최사 규정상 **대회 데이터와 베이스라인 코드는 공개 금지** → git에서 완전히 제외.

## 작업 환경 (로컬 ↔ git ↔ 서버)

- **로컬PC**: 모든 코드 작성/수정/디버깅. `git add / commit / push`.
- **GitHub**: 코드 동기화 및 백업 (private repo).
- **서버PC (RTX 3090)**: 학습/추론 실행 전용. `git pull`만, 코드 수정 금지.

### 로컬PC 디렉토리
```
~/projects/commerce-recsys-cy/        # git repo
├── CLAUDE.md, .gitignore, README.md
├── shared/, members/cy/, submissions/
├── (선택) data/, code/                # 다운로드 후 .gitignore로 제외됨
└── output/                            # 로컬 검증용 (.gitignore)
```

### 서버PC 디렉토리
```
/root/
├── data/                              # 원본 데이터 (주최사, 공개 금지)
│   ├── train.parquet
│   ├── sample_submission.csv
│   └── (생성됨) user2idx.json, item2idx.json, SASRec_dataset/
├── code/                              # 베이스라인 코드 (주최사, 공개 금지)
│   ├── train_als.py, train_sasrec.py, ...
└── workspace/
    └── recsys-commerce/               # ★ git repo (clone)
        ├── CLAUDE.md, .gitignore, README.md
        ├── shared/                    # 공용 유틸 (자체 작성)
        ├── members/cy/                # 개인 실험
        │   └── exp_NNN_<name>/
        │       ├── train.py, inference.py, config.yaml, README.md  (★ git)
        │       ├── predictions.parquet, output.csv                  (.gitignore)
        │       └── saved/                                           (.gitignore)
        └── submissions/log.md         # 제출 이력
```

**경로 참조 규칙** (학습 스크립트):
- 데이터: 절대경로 `/root/data/train.parquet`
- 베이스라인: 절대경로 `/root/code/` (참고용, **복사 금지**)
- 산출물: 상대경로 `./predictions.parquet`, `./saved/`, `./output.csv` (실험 폴더 기준)

## 대회 핵심
- **태스크**: train 데이터의 모든 user(**638,257명**)에게 각 10개씩 item 추천
- **평가**: **NDCG@10 (binary relevance)** — public/private 50:50 random split
- **동점이면 제출 횟수가 적은 쪽이 상위** → 무의미한 제출 피하기, 자체 validation 우선
- **베이스라인 점수**: ALS = 0.0847 / SASRec = 0.0842

## 환경 (서버PC)
- Ubuntu 20.04.6, **RTX 3090 24GB**, 64 cores, **251GB RAM**, 1.8TB disk
- conda `base`: **Python 3.10.13, PyTorch 2.1.0 (CUDA 12.2)**, pandas 2.1.4, numpy 1.26.0
- 베이스라인 의존성: `recbole`, `kmeans_pytorch`, `ray`, `implicit`, `pyarrow`, `fastparquet`, `tqdm`
- 단일 사용자 / 단일 GPU. 메모리 여유 충분해서 train.parquet 전부 로드 OK.

⚠️ **대회 기간 종료 시 서버 자체가 회수됨**. 종료 전 모든 자산을 git/wandb/로컬에 백업 완료해야 함.

---

## ⚠️ 주최사 공개 금지 자산

다음은 **모두 `.gitignore` 적용**되며 **public 전환 시에도 절대 노출 금지**:

| 자산 | 위치 | 사유 |
|---|---|---|
| 대회 데이터 | `/root/data/`, 로컬 `data/` | 주최사 재배포 금지 |
| 베이스라인 코드 | `/root/code/`, 로컬 `code/` | 주최사 재배포 금지 |
| 데이터 파생물 | `user2idx.json`, `item2idx.json`, `SASRec_dataset/` 등 | 원본 ID 포함 |
| 모델 weights, prediction csv/parquet | `members/*/exp_*/` 산출물 | 데이터 정보 포함 가능 |
| 압축 파일 | `*.tar.gz`, `*.zip` | 원본 데이터/코드 포함 가능 |

**주의**:
- **베이스라인 코드를 그대로 복사해 새 실험에 넣지 말 것**. 컨벤션/패턴은 참고하되 실제 구현은 본인이 새로 작성.
- 대회 종료 후 public 전환 시 git history도 점검 (실수로 들어간 데이터/코드 있는지).

---

## 데이터 스키마 (train.parquet, 8.35M × 8)
| 컬럼 | dtype | 비고 |
|---|---|---|
| `user_id` | str (UUID) | 638,257 unique |
| `item_id` | str (UUID) | 29,502 unique |
| `user_session` | str | 사용자가 오래 쉬면 갱신됨 |
| `event_time` | **object (string!)** | UTC, format `'%Y-%m-%d %H:%M:%S %Z'` |
| `category_code` | str | nullable |
| `brand` | str | nullable |
| `price` | float64 | |
| `event_type` | str | `view` / `cart` / `purchase` |

**기간**: train = 2019-11-01 ~ 2020-02-29 (4개월) / 평가 = 2020-03-01 ~ 2020-03-07 (1주).
평가 데이터는 train에 등장한 user/item으로만 구성됨 (cold-start 신규 ID 없음).

## 제출 형식 (output.csv)
- 컬럼: `user_id,item_id` (헤더 포함)
- **정확히 6,382,570 rows** (638,257 user × 10)
- 각 user당 **score 내림차순**으로 정렬된 **서로 다른** 10개 item

---

## 표준 산출물 (모든 실험 공통)

각 실험은 다음 3가지를 반드시 산출:

### 1. `predictions.parquet` — **앙상블 표준 입력**
top-50 + score. 대회 사이트 submission.csv(top-10만, score 없음)는 앙상블에 부족.

```python
columns:
  - user_id: str (원본 UUID)
  - item_id: str (원본 UUID)
  - score:   float (모델 출력 raw score)
  - rank:    int (1~50, user 내 순위)
shape: 638,257 × 50 = 31,912,850 rows
```

### 2. `output.csv` — 제출용
`predictions.parquet` → `shared/submission.py`로 변환 (dedup + popularity fallback + 형식 검증).

### 3. wandb artifact — 백업
- 모델 weights: `cy_exp_NNN_<model>` (type: model)
- predictions: `cy_exp_NNN_predictions` (type: prediction)

**왜 top-50인가**: 앙상블 시 후보 다양성 확보 + reranker 학습 여지.

---

## 자주 하는 실수 (gotchas)
1. **638,257명 전원 채우기 필수** — cold-start user는 popularity fallback.
2. **유저당 item 중복 금지** — dedup 후 부족하면 popularity로 채움.
3. **`event_time`은 string** — `pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')` 변환.
4. **`category_code`, `brand`에 결측 존재**.
5. **user/item ID가 UUID** — 학습은 정수 idx 매핑 후, 출력 시 역매핑.
6. **시간 누수 주의** — 자체 validation은 마지막 1주를 hold-out.
7. **NDCG 정렬 방향** — score 큰 것이 첫 row.
8. **베이스라인 코드 복사 금지** — 패턴만 참고, 구현은 새로 작성.

## 베이스라인 컨벤션 (참고만)
- `argparse` 기반 CLI
- `set_seed(42)` 항상 호출 (`shared/utils.py`에 자체 작성)
- 데이터 절대경로 `/root/data/` 사용

---

## Git 워크플로우

**기본 패턴**: 로컬 작성 → push → 서버 pull → 학습.

### 브랜치 + 폴더 하이브리드
- **브랜치**: 큰 방향성 변경 시에만 (`exp/lightgcn`, `exp/two-stage`)
- **폴더**: 같은 모델군 변형은 `members/cy/exp_NNN_<name>/`로 분리
- `main`: 베이스라인 + 검증된 개선
- `exp/<name>`: 실험 브랜치 → main에 merge

### 커밋 규칙
- 의미 있는 변경마다 push (서버 다운/회수 대비)
- 메시지: `[exp_NNN] <설명>`, `[shared] <설명>`, `[doc] <설명>`
- 각 실험 `README.md`에 가설/하이퍼/점수 기록
- **커밋 전 `git status`로 데이터/베이스라인 파일이 staging에 없는지 확인**

### 서버에서 절대 하지 말 것
- git tracking 파일(`shared/`, `members/cy/`의 .py/.md) 수정
- 만약 임시 수정이 필요하면: `git stash` → `git pull` → `git stash pop`

---

## 백업 3중화

| 자산 | 1차 저장소 | 2차 백업 |
|---|---|---|
| 본인 코드 | 서버 `/root/workspace/recsys-commerce/` | **GitHub** |
| 베이스라인 코드 | 서버 `/root/code/` | 주최사 URL 재다운로드 |
| 모델 weights | `members/cy/exp_NNN/saved/` | **wandb artifact** |
| predictions.parquet | `members/cy/exp_NNN/` | **wandb artifact** |
| 제출 csv | `members/cy/exp_NNN/output.csv` | **로컬 다운로드 + wandb log** |
| 실험 메트릭 | wandb (클라우드) | — |
| 원본 데이터 | `/root/data/` | **로컬PC 다운로드 + 주최사 URL** |

### 학습 스크립트 wandb 백업 템플릿
```python
import wandb

wandb.init(project='cy-commerce-recsys', name=f'cy_exp_{NNN}_{model_name}', config=config)

wandb.log({'epoch': epoch, 'train_loss': loss, 'val_ndcg10': ndcg})

# 학습 종료 후
artifact = wandb.Artifact(f'cy_exp_{NNN}_{model_name}', type='model')
artifact.add_file('./saved/best_model.pth')
wandb.log_artifact(artifact)

pred_artifact = wandb.Artifact(f'cy_exp_{NNN}_predictions', type='prediction')
pred_artifact.add_file('./predictions.parquet')
wandb.log_artifact(pred_artifact)
```

---

## 실험 워크플로우
1. **로컬PC**: `members/cy/exp_NNN_<name>/` 폴더 생성 + 코드 작성
2. `shared/` 함수 사용 (data_loader, metrics, validation, submission)
3. `git status` 확인 후 `git commit && git push`
4. **서버PC**: `cd /root/workspace/recsys-commerce && git pull`
5. `cd members/cy/exp_NNN_<name> && python train.py` (wandb 백업 포함)
6. 자체 validation NDCG@10 확인 → inference 진행
7. `predictions.parquet` → `shared/submission.py`로 `output.csv` 변환
8. 형식 검증 후 제출
9. **로컬PC**: 결과를 `experiments README` + `submissions/log.md`에 기록 → push

**제출 횟수 절약 원칙**: 자체 validation에서 베이스라인 대비 명확히 더 좋을 때만 제출.

---

## 팀 합류 시 이전 절차 (미래 대비)

1. `members/cy/` 폴더 통째로 팀 repo의 `members/cy/`로 복사
2. `shared/` 호환성 점검
3. wandb: 개인 project → 팀 project로 run 이동
4. `submissions/log.md` 팀 로그에 병합
5. brnach 정리

**지금부터 지킬 것**: `members/cy/` 구조 유지, `predictions.parquet` 표준, `shared/` 일반성, README 일관성.

---

## Public 전환 직전 체크리스트 (대회 종료 후)

```bash
# 1. 데이터/베이스라인이 git history에 들어간 적 없는지
git log --all --oneline -- '*.parquet' '*.csv' 'data/' 'code/'

# 2. 민감 정보 패턴
git log -p | grep -iE '(api_key|password|secret|token|@.*\.com)' | head

# 3. 큰 파일 검색
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print $3, $4}' | sort -rn | head -20
```

문제 있으면 `git filter-repo`로 history 제거. README 정비, 라이선스 결정.

---

## 자주 쓰는 명령어

### 서버에서
```bash
# git 동기화
cd /root/workspace/recsys-commerce && git pull

# 베이스라인 ALS (약 30초, 환경 검증용)
cd /root/code && python train_als.py

# 데이터 sanity check
python -c "import pandas as pd; df = pd.read_parquet('/root/data/train.parquet'); print(df.shape, df['user_id'].nunique(), df['item_id'].nunique())"

# 실험 실행
cd /root/workspace/recsys-commerce/members/cy/exp_NNN_<name>
python train.py
python inference.py

# 제출 파일 검증 (제출 전 필수)
python -c "
import pandas as pd
sub = pd.read_csv('./output.csv')
assert sub.shape == (6382570, 2), f'shape: {sub.shape}'
assert sub['user_id'].nunique() == 638257, 'user count mismatch'
assert (sub.groupby('user_id').size() == 10).all(), 'not 10 per user'
assert sub.groupby('user_id')['item_id'].apply(lambda x: x.is_unique).all(), 'duplicate items per user'
print('OK')
"
```

### 로컬에서
```bash
# 제출 csv 백업 (서버에서 로컬로)
scp <user>@<server>:/root/workspace/recsys-commerce/members/cy/exp_NNN/output.csv ./submissions/

# 결과 기록 후 push
cd ~/projects/commerce-recsys-cy
git add members/cy/exp_NNN/README.md submissions/log.md
git commit -m "[exp_NNN] results: val 0.0892 / public 0.0865"
git push
```

---

## 시도 후보
- **MF/Implicit**: ALS, BPR, LightGCN, EASE, RecVAE
- **Sequential**: SASRec(베이스라인), BERT4Rec, GRU4Rec, SR-GNN
- **Graph**: LightGCN, NGCF, UltraGCN
- **Two-stage**: candidate generation(ALS/co-visit) + reranker(LightGBM/CatBoost)
- **Item2Vec/Prod2Vec**, item-item co-visitation
- **Feature engineering**: recency, frequency, session sequence, brand/category affinity, price band, event_type 가중치
- **앙상블**: Reciprocal Rank Fusion (RRF) 또는 가중 평균

---

## Claude Code 작업 가이드라인
- 새 실험은 `members/cy/exp_NNN_<name>/`에 작성. **베이스라인 `/root/code/`는 수정/복사하지 않음**.
- **`shared/` 함수 적극 활용** — data_loader, metrics, validation, submission. 없으면 만들어서 추가.
- 새 학습 스크립트엔 **wandb 통합 + artifact 백업 코드 기본 포함**.
- **모든 실험은 `predictions.parquet` (top-50 + score) 산출 필수**.
- 제출 파일 만들면 **위 검증 스크립트로 형식 확인 후 보고**.
- 큰 변경 전엔 자체 validation NDCG@10 보고 후 진행.
- 메모리 251GB 여유 → parquet 전체 로드 OK. user-item matrix는 `scipy.sparse.csr_matrix`.
- GPU 24GB 단일 → SASRec/BERT4Rec batch_size 4096 정도까지.
- 새 데이터/큰 산출물 생성 시 **`.gitignore` 패턴 확인**.
- 실험 폴더 만들 때 `README.md`도 함께 — 가설/하이퍼/점수/결론.
- `shared/` 코드는 **cy 환경 하드코딩 금지** (팀 합류 대비).
- ⚠️ **주최사 데이터/베이스라인 코드를 git에 올리지 말 것** — staging 전 `git status` 확인.
- ⚠️ **베이스라인 코드 그대로 복사 금지** — 컨벤션/패턴만 참고.
- ⚠️ **서버에서 코드 수정 금지** — 로컬에서만 작성, 서버는 `git pull`만.
