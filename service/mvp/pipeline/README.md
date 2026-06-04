# pipeline/ — 실데이터 연결 파이프라인 (Phase 2~3)

| 항목 | 내용 |
| ---- | ---- |
| 최종 수정 | 2026-05-28 |
| 역할 | train.parquet + submission CSV → alias·프로필·Evidence Pack 재빌드 |
| 전제 | Phase 1 완료 (`side_project/` 전체 동작 확인) |
| 실행 위치 | `side_project/` 루트 (`cd side_project`) |

---

## 전체 4단계 흐름

```
Phase 1 ✅ 완료          Phase 2                Phase 3              Phase 4 (선택)
─────────────────     ─────────────────     ─────────────────     ─────────────────
mock 데이터로          실데이터 전처리         4종 추천 엔진          LangGraph 멀티턴
전체 파이프라인         id_alias.py            recommenders.py       langgraph_advisor.py
동작 확인              user_profile.py        실데이터 연결
                       build_rag_index.py
```

---

## Phase 1 — LLM 레이어 데모 ✅ 완료

본체 없이 mock 데이터만으로 전체 파이프라인이 동작하는 상태.
팀원이 받으면 아래 명령만 실행하면 바로 데모 확인 가능.

```bash
cd side_project
pip install -r requirements.txt

# Evidence Pack 빌드 (mock)
python -m evidence_pack.builder --adapter mock --out data/evidence_pack.jsonl

# Streamlit 데모 실행
streamlit run ui/app.py
```

포함된 것:
- Evidence Pack (38+ 신호, revisit 포함)
- Trust Gate 3중 검증 (Hard-gate → SelfCheckGPT → Calibration)
- 4종 추천 유형 (submission / content / unseen / revisit)
- Streamlit UI (사용자 탭 + 운영자 탭)
- LLM-as-Judge 오프라인 평가

---

## Phase 2 — 실데이터 연결 (ID 별칭 + SQLite 프로필)

**입력**: `../data/train.parquet`, `../outputs/submission_reranker_lgbm.csv`  
**출력**: `data/id_aliases.json`, `data/user_profiles.db`

### 실행 순서

```bash
# (모두 side_project/ 루트에서 실행)

# Step 1: UUID ↔ alias 양방향 매핑 생성
python -m pipeline.id_alias \
  --train ../data/train.parquet \
  --submission ../outputs/submission_reranker_lgbm.csv \
  --out data/id_aliases.json
# → users: 638,257  items: 29,502  → data/id_aliases.json

# Step 2: 유저별 이벤트 집계 → SQLite
python -m pipeline.user_profile \
  --train ../data/train.parquet \
  --aliases data/id_aliases.json \
  --out data/user_profiles.db
# → Wrote 638,257 user profiles → data/user_profiles.db

# Step 3: alias 주입된 Evidence Pack 재빌드
python -m pipeline.build_rag_index \
  --aliases data/id_aliases.json \
  --submission ../outputs/submission_reranker_lgbm.csv \
  --train ../data/train.parquet \
  --out data/evidence_pack.jsonl
# → Wrote N evidence packs → data/evidence_pack.jsonl

# Step 4: UI 확인 (별도 수정 없음)
streamlit run ui/app.py
```

### 검증

```bash
python -c "
import json, sqlite3

a = json.load(open('data/id_aliases.json'))
print('users:', len(a['users']), '/ items:', len(a['items']))
# 샘플 확인
uid = next(iter(a['users']))
print('user sample:', uid, '->', a['users'][uid])

con = sqlite3.connect('data/user_profiles.db')
n = con.execute('SELECT COUNT(*) FROM user_profiles').fetchone()[0]
print('profiles:', n)
row = con.execute('SELECT user_alias, seen_items FROM user_profiles LIMIT 1').fetchone()
print('seen_items sample:', row[0], json.loads(row[1])[:3])
"
```

### 파일별 스펙

#### `id_alias.py` — `build_aliases(train_parquet, submission_csv) → dict`

출력 포맷:
```json
{
  "users":     {"<uuid>": "user_00001", ...},
  "users_inv": {"user_00001": "<uuid>", ...},
  "items":     {"<uuid>": "shoes.keds.kapika.mid_0001", ...},
  "items_inv": {"shoes.keds.kapika.mid_0001": "<uuid>", ...}
}
```

Item Semantic ID 규칙: `{category_l2}.{category_l3}.{brand_slug}.{price_bucket}_{seq:04d}`

| 구성 요소 | 설명 | 예시 |
|---|---|---|
| category_l2 | category_code 앞 2파트 | `apparel.shoes` |
| category_l3 | category_code 앞 3파트 | `apparel.shoes.keds` |
| brand_slug | 브랜드명 소문자 + 하이픈 | `kapika` |
| price_bucket | low(<30) / mid(30~80) / high(>80) | `mid` |
| seq | 같은 bucket 내 순번 | `0001` |

#### `user_profile.py` — `build_profiles(train_parquet, aliases) → list[dict]`

SQLite `user_profiles` 테이블:

| 컬럼 | 타입 | 내용 |
|---|---|---|
| `user_alias` | TEXT PK | `user_00001` |
| `user_id` | TEXT | 원본 UUID |
| `total_events` | INTEGER | 전체 이벤트 수 |
| `recent14_events` | INTEGER | max(event_time) 기준 최근 14일 수 |
| `top_categories` | JSON array | view 기반 top-3 category_l2 |
| `top_brands` | JSON array | view 기반 top-3 brand |
| `carted_brands` | JSON array | cart 이벤트 brand (unique) |
| `purchased_brands` | JSON array | purchase 이벤트 brand (unique) |
| `price_median` | REAL | view 기준 가격 중앙값 |
| `price_p25` | REAL | view 기준 1사분위 |
| `price_p75` | REAL | view 기준 3사분위 |
| `price_iqr` | REAL | p75 - p25 |
| `seen_items` | JSON array | view+cart+purchase item_alias 통합 |
| `carted_items` | JSON array | cart item_alias 목록 |
| `purchased_items` | JSON array | purchase item_alias 목록 |

집계 기준: `max(event_time)` 기준 최근 14일 / 전체 기간 구분  
recent14 cutoff = `max_time - 14days`

#### `build_rag_index.py` — `build_evidence_pack_with_alias(...) → int`

1. `CSVAdapter` + `ParquetCatalogSource`로 UUID 기준 Evidence Pack 빌드
2. 각 `EvidencePack`의 `user_alias`, `item_alias` 필드를 aliases로 주입
3. `data/evidence_pack.jsonl` 덮어씀 → `streamlit run ui/app.py` 그대로 실데이터 반영

```
(선택) --faiss 플래그:
  출력: data/user_neighbors.faiss + data/user_neighbors_meta.pkl
  용도: Phase 3 CF 기반 유사 유저 추천
  의존성: pip install faiss-cpu scipy
```

---

## Phase 3 — 4종 추천 엔진 실연결

**전제**: Phase 2 완료 (`data/id_aliases.json`, `data/user_profiles.db` 존재)  
**파일**: `pipeline/recommenders.py`

### 4종 엔진 인터페이스

```python
from pipeline.recommenders import (
    recommend_submission,
    recommend_profile_content,
    recommend_unseen_recency,
    recommend_revisit,
)
```

| 함수 | 풀 | 반환 |
|---|---|---|
| `recommend_submission(user_alias, recs, top_n)` | submission top-K | `list[str]` item_alias |
| `recommend_profile_content(user_alias, db_path, catalog, submission_top, top_n)` | unseen | `list[str]` |
| `recommend_unseen_recency(user_alias, db_path, catalog, recency_items, submission_top, top_n)` | unseen | `list[str]` |
| `recommend_revisit(user_alias, db_path, event_log, top_n)` | seen | `list[dict]` (revisit_score 포함) |

유형 1·2·3 = unseen 중심, 유형 4(revisit) = seen 중심 → 풀 충돌 없음

### 점수화 로직

**content 추천** (`recommend_profile_content`):
```
+2: category_l2가 profile.top_categories 안에 있음
+2: brand가 profile.top_brands 또는 carted_brands 안에 있음
+1: price가 price_median ± 1.5 × price_iqr 범위 안
+1: submission_top 앞쪽 아이템 보너스
fallback: 점수 0이면 unseen + submission 순위만
```

**unseen recency 추천** (`recommend_unseen_recency`):
```
unseen × recency_items 교집합
category_l2 ∈ top_categories 우선
submission 순위 보너스
fallback: unseen+카테고리 → unseen만
```

**revisit 추천** (`recommend_revisit`):
```
evidence_pack.builder.recommend_revisit() 직접 호출
event_log 포맷: [{"item_id": item_alias, "event_type": ..., "days_ago": int}]
7일 이내 purchase 아이템 자동 제외
TIFU 스타일: event weight × temporal decay × repeat bonus
```

### `evidence_pack/adapter.py` — `AliasCSVAdapter`

Phase 2 완성 후 alias 기반 submission을 Evidence Pack 빌더에 직접 연결하는 어댑터.

```python
from evidence_pack.adapter import AliasCSVAdapter, ParquetCatalogSource
from evidence_pack.builder import build_evidence_pack

rec = AliasCSVAdapter(
    submission_csv="../outputs/submission_reranker_lgbm.csv",
    aliases_json="data/id_aliases.json",
    top_k=10,
)
cat = ParquetCatalogSource("../data/train.parquet")
build_evidence_pack(rec, cat, "data/evidence_pack.jsonl")
```

---

## Phase 4 — Streamlit 챗봇 고도화 (선택)

**파일**: `ui/langgraph_advisor.py`  
**의존성**: `pip install langgraph langchain-core`

### LangGraph 멀티턴 (`AdvisorGraph`)

Phase 3까지는 버튼 클릭 → 단발 응답.
Phase 4에서 `AdvisorGraph`로 교체하면 "더 저렴한 걸로", "다른 브랜드는?" 같은 연속 대화가 가능해짐.

```python
# ui/app.py 교체 포인트
from ui.langgraph_advisor import AdvisorGraph

graph = AdvisorGraph(pack, selected_item_id=target_item)
resp = graph.chat("더 저렴한 대안 있어?", thread_id=pack.user_id)
```

그래프 구조:
```
사용자 메시지
    │
    ▼
node_classify_intent          # "shopping" / "general" 분류
    │
    ├── shopping → node_shopping   # Evidence Pack + trust_gate
    │
    └── general  → node_general    # Solar Pro 직접 (⚠️ 미검증)
```

상태: `AdvisorState` (pack, messages, selected_item_id, intent, last_response)  
세션 기억: `MemorySaver` — 같은 `thread_id`면 대화 맥락 유지

### UI accordion 분리 (app.py 수정)

```python
with st.expander("📋 추천 상품 (submission 기반)", expanded=True):
    for rec in submission_recs: recommendation_card(rec)
with st.expander("🎯 취향 매칭", expanded=False):
    for rec in content_recs: recommendation_card(rec)
with st.expander("🆕 신상품 / 아직 못 본 상품", expanded=False):
    for rec in recency_recs: recommendation_card(rec)
with st.expander("↩ 다시 볼 상품", expanded=False):
    for rec in revisit_recs: recommendation_card(rec)
```

---

## 파일 구조

```
pipeline/
  README.md             이 파일
  __init__.py
  id_alias.py           Phase 2: UUID ↔ alias 매핑 생성
  user_profile.py       Phase 2: 유저 프로필 집계 → SQLite
  build_rag_index.py    Phase 2: alias 주입 Evidence Pack 재빌드 + (선택) FAISS
  recommenders.py       Phase 3: 4종 추천 엔진
```

관련 파일:
- [`evidence_pack/adapter.py`](../evidence_pack/adapter.py) — `AliasCSVAdapter` (Phase 3 연결)
- [`ui/langgraph_advisor.py`](../ui/langgraph_advisor.py) — Phase 4 멀티턴 스켈레톤

---

## 수정·확장 포인트

| 하고 싶은 것 | 어디서 |
|---|---|
| price_bucket 구간 조정 | `id_alias.py` `_price_bucket()` |
| 프로필 집계 기간 변경 (14일 → 7일 등) | `user_profile.py` `build_profiles()` cutoff |
| content 추천 점수 가중치 조정 | `recommenders.py` `_score()` 내 +2/+1 값 |
| unseen recency 기준 아이템 목록 변경 | `build_rag_index.py` 또는 호출부에서 `recency_items` 생성 |
| FAISS CF 후보 추가 | `build_rag_index.py` `build_faiss_index()` + `ItemSignals`에 cf_neighbor 필드 추가 |
| LangGraph 노드 추가 | `ui/langgraph_advisor.py` `node_*` 함수 추가 |
