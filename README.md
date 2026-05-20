# Commerce Behavior Purchase Prediction

부트캠프 RecSys 경진대회. 4개월 이커머스 행동 로그로 다음 1주일 구매 아이템 top-10을 예측하는 추천 시스템.

## 작업 구조

- **로컬PC**: 모든 코드 작성/수정. `git add / commit / push`.
- **서버PC (RTX 3090)**: 학습/추론 실행 전용. `git pull`만.

## 프로젝트 구조

```
.
├── CLAUDE.md          # Claude Code 작업 컨텍스트
├── .gitignore
├── README.md
├── shared/            # 공용 유틸 (data_loader, metrics, validation, submission, ensemble)
├── members/cy/        # 개인 실험 폴더
│   └── exp_NNN_<name>/
└── submissions/log.md # 제출 이력
```

자세한 컨벤션과 가이드라인은 `CLAUDE.md` 참고.

## ⚠️ 공개 금지 자산

본 repo는 다음을 **포함하지 않음** (주최사 규정):
- 대회 데이터 (`train.parquet`, `sample_submission.csv`)
- 주최사 제공 베이스라인 코드 (`/code/`)
- 학습 산출물, prediction 파일

## 재현 절차 (데이터/베이스라인 다운로드)

본 repo의 코드를 실행하려면 주최사 제공 데이터와 베이스라인을 별도 다운로드해야 합니다.

### 서버PC 셋업
```bash
# 1. 데이터 (/root/data/)
mkdir -p /root/data && cd /root/data
wget <주최사 제공 data URL>
tar -xzvf data.tar.gz

# 2. 베이스라인 코드 (/root/code/)
cd /root
wget <주최사 제공 code URL>
tar -xzvf code.tar.gz

# 3. 의존성 설치
pip install -r /root/code/requirements.txt
pip install wandb && wandb login

# 4. git repo clone
mkdir -p /root/workspace && cd /root/workspace
git clone <REPO_URL>

# 5. 실험 실행
cd recsys-commerce/members/cy/exp_NNN_<name>
python train.py
```

### 로컬PC 셋업
```bash
git clone <REPO_URL> ~/projects/commerce-recsys-cy
cd ~/projects/commerce-recsys-cy

# (선택) 데이터/베이스라인 로컬 다운로드 — 종료 후 보존용
mkdir data && cd data
wget <주최사 제공 data URL> && tar -xzvf data.tar.gz
cd .. && wget <주최사 제공 code URL> && tar -xzvf code.tar.gz
# (.gitignore로 자동 제외됨)
```

다운로드 URL은 부트캠프 대회 페이지에서 확인.

## 환경

- Ubuntu 20.04, Python 3.10, PyTorch 2.1 (CUDA 12.2), RTX 3090 24GB

## License

TBD (대회 종료 후 결정)
