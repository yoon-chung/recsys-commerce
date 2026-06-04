# eval/ — 오프라인 회귀 채점 (LLM-as-Judge)

## 무엇인가

Golden set에 대해 LLM-as-Judge로 응답 품질을 1~5로 채점한다. **런타임이 아닌 회귀 지표** — 프롬프트나 모델, 게이트, Evidence Pack 스키마를 바꿨을 때 품질이 떨어졌는지 자동 감지하는 용도.

차원 (G-Eval 풍):
- **groundedness** — 응답의 모든 주장이 Evidence Pack 신호로 뒷받침되는가
- **relevance** — 질문(question_key)에 정확히 답했는가
- **actionability** — 사용자/운영자가 다음 행동을 결정하는 데 도움이 되는가

## 왜 필요한가

Hard-gate / SelfCheck / Calibration은 **개별 응답**을 거른다. 하지만 "전체 시스템 품질이 좋아졌나 나빠졌나"는 **golden set 집계**로만 알 수 있다.

또 Calibration의 라벨(`label_supported`)도 여기서 관리된다 — 즉 이 폴더가 trust_gate를 **학습**시키는 데이터를 보관한다.

## 추천시스템·LLM과 어떻게 연결되나

```
evidence_pack.jsonl + advisor 응답
        │
        ▼
  golden_set.jsonl ──▶ judge.score (Mock or LLM) ──▶ JudgeScore × N
        │                                                  │
        ▼                                                  ▼
  Calibrator.fit(label_supported)                   regression_run 집계
        │                                                  │
        ▼                                                  ▼
  trust_gate.calibration                            품질 회귀 알람
```

Golden set의 `pack`은 현재 Evidence Pack 스키마를 그대로 저장한다. 가격대, brand/category affinity, conversion/trend 같은 새 근거 신호도 judge 입력에 포함된다.

## 핵심 파일

| 파일 | 역할 |
|---|---|
| [`golden_set.py`](golden_set.py) | `GoldenItem` 스키마 + JSONL 로더/세이버 + `make_starter_golden_set()` (mock으로 시드 생성) |
| [`judge.py`](judge.py) | `JudgeScore`, `MockJudge`, `LLMJudge`, `get_judge()`, `regression_run()` |
| [`__init__.py`](__init__.py) | lazy export. `python -m eval.judge` 단독 실행 경고 방지 |

## 사용 예시

### 1) Starter golden set 만들기 (개발 시작용)
```bash
python -m eval.golden_set --n 10 --out data/golden_set.jsonl
# → 10건이 mock으로 만들어짐. 모두 label_supported=True로 시드.
#   각 항목을 팀이 직접 검토해 label을 수정/보완.
```

### 2) 회귀 채점 돌리기
```bash
python -m eval.judge --golden data/golden_set.jsonl
# mock judge 또는 (API 키 있으면) solar-pro judge로 자동 채점
```

Golden set이 없을 때 starter를 같이 만들려면:
```bash
python -m eval.judge --golden data/golden_set.jsonl --make-starter --n-starter 10
```

출력 예:
```json
{
  "n": 10,
  "judge": "mock-judge",
  "groundedness_mean": 5.0,
  "relevance_mean": 4.6,
  "actionability_mean": 4.4,
  "overall_mean": 4.67,
  "labeled_supported_frac": 1.0
}
```

### 3) 파이썬 코드로
```python
from eval import iter_golden_set, get_judge, regression_run

items = list(iter_golden_set("data/golden_set.jsonl"))
result = regression_run(items)  # judge 자동 선택
print(result["overall_mean"], result["groundedness_mean"])
```

### 4) Calibrator 학습 (라벨 활용)
```python
from eval import iter_golden_set
from trust_gate import Calibrator

items = list(iter_golden_set("data/golden_set.jsonl"))
confs = [it.response.confidence_raw for it in items]
labels = [1.0 if it.label_supported else 0.0 for it in items]

cal = Calibrator().fit(confs, labels)
cal.save("data/calibrator.json")
# UI는 자동으로 이 calibrator.json을 읽어 사용
```

## Bias 완화 (LLM-as-Judge)

| Bias | 우리 방어 |
|---|---|
| Position bias | absolute scoring이라 영향 작음. pairwise 비교 시 순서 swap 후 평균. |
| Verbosity bias | 시스템 프롬프트에 "길이로 점수 주지 말 것" 명시. |
| Self-enhancement bias | 가능하면 judge LLM과 advisor LLM을 다른 모델로 (다른 vendor 권장). |
| Evidence over-trust | hard_gate 통과만으로 만점이 아님. relevance/actionability를 별도 축으로 봄. |

## 수정·확장 포인트

| 하고 싶은 것 | 어디서 |
|---|---|
| 새 차원 추가 (예: safety) | `judge.py` `JudgeScore` + `JUDGE_SYSTEM_PROMPT` + `MockJudge.score` + `regression_run` |
| Pairwise A/B 비교 | `judge.py`에 `pairwise_judge(resp_a, resp_b)` 추가, 순서 swap 평균 |
| 다른 judge LLM | `LLMJudge.__init__`에 base_url/모델 다르게 |
| 라벨링 도구 | (선택) Streamlit 별도 페이지로 GoldenItem 보고 label_supported를 토글하는 UI |
| 새 quick question 커버 | `golden_set.py`의 `questions` 리스트 갱신 |

## 조정 다이얼 (튜닝 포인트)

골든셋 규모와 채점 차원/임계. "↑/↓하면 무슨 일이 일어나는지" 명시.

### `golden_set.py` — Starter 생성
| 위치 | 기본 | 의미 | 조정 가이드 |
|---|---|---|---|
| `make_starter_golden_set(n=)` | 10 | starter 항목 수 | 데모는 10, **calibration·회귀에 의미 있으려면 40+** 권장 |
| `make_starter_golden_set` 내 `questions` 리스트 | 6종 (`why`/`should_buy`/`fit_preference`/`revisit`/`category_trend`/`promotion_candidate`) | starter가 커버할 질문 다양성 | `advisor.QUICK_QUESTIONS`의 신규 키 추가 시 함께 갱신 |
| `make_starter_golden_set` 내 `top_k=10`, `seed=42` | (10, 42) | MockAdapter 인자 | 본체 top-K와 일치 |
| `label_supported=True` (모두) | True | starter 라벨 | **팀이 직접 검토·수정 필수** — 안 하면 calibration이 saturate |
| `tmp_pack.unlink(missing_ok=True)` | 임시 pack 삭제 | starter 생성 후 찌꺼기 제거 | 유지 권장 |

### `judge.py` — 채점 차원·점수 분기
| 위치 | 기본 | 의미 | 조정 |
|---|---|---|---|
| `JudgeScore` 필드 | 3차원 (`groundedness` / `relevance` / `actionability`), 각 1~5 | 채점 축 | 차원 추가 시 필드+프롬프트+`MockJudge.score`+`regression_run` 4곳 동기화 |
| `JUDGE_SYSTEM_PROMPT` 규칙 | "길이로 점수 주지 말 것" 등 | bias 완화 지시 | verbosity bias 더 엄격히 하려면 추가 |
| `MockJudge.score` `g` (groundedness) | 5 (pass), 1 (fail), wobbly flag당 -1 | mock 채점 분기 | 실 LLM 채점과의 align 비교 후 보정 |
| `MockJudge.score` `r` (relevance) | claims ≥ 3 → 5, ≥ 2 → 4, == 0 → 1 | claims 갯수 ↔ relevance 매핑 | 너무 관대하면 회귀 감지력↓ |
| `MockJudge.score` `a` (actionability) | decision_hint 존재 시 4, + cal confidence > 0.5 시 5 | actionability 분기 | confidence 의존도 조정 |
| `LLMJudge` `temperature=0.0` | 0.0 | 채점의 결정성 | **0이 표준** — 채점 재현성. 올리면 회귀 감지 노이즈↑ |

### `regression_run` — 집계
| 위치 | 기본 | 의미 | 조정 |
|---|---|---|---|
| `statistics.mean(...)` over rows | mean | 차원별 집계 | median으로 바꾸면 outlier에 robust |
| `labeled_supported_frac` | sum(label==True) / N | golden set 양성 비율 | 정보용 — calibration 라벨 쏠림 확인 |
| `rows` 상세 | user/item/question별 점수 | 실패 표본 추적 | UI 리포트로 확장 가능 |

### Calibrator 학습 (이 폴더에서 직접 호출)
| 위치 | 기본 | 의미 | 조정 |
|---|---|---|---|
| `labels = [1.0 if it.label_supported else 0.0 ...]` | binary | calibration 타겟 | judge `overall >= 4` 같이 "품질 점수 기반"으로 바꿀 수도 있음 |
| 학습 후 ECE before/after 비교 권장 | — | 보정 효과 검증 | T가 saturate되면 (T<0.1 또는 T>5) golden set 다양성 부족 신호 |

> **회귀 감지의 의미 단위**: 일반적으로 ECE는 절대값 0.05 이상 변화 / judge overall은 0.3+ 변화가 "유의미한 회귀"의 1차 임계. 표본이 적으면 신뢰구간이 넓으니 40건 이하에선 직관적 점수 차이 정도만 본다.

---

## 한계 / 주의

- LLM-Judge 점수는 **절대 점수 아닌 상대비교/회귀 감지**용. 0.1 차이 추격에 의미 두지 말 것.
- starter golden set은 mock 기반 → label이 모두 True로 들어감. **팀이 직접 검토하여 label을 수정**해야 calibration 학습이 의미 있어진다.
- judge LLM과 advisor LLM이 같으면 self-enhancement bias로 점수가 과대평가될 수 있음 — 다른 vendor 권장.
- 표본이 작으면(<20) 평균의 분산이 큼. 회귀 감지엔 40건 이상 권장.

---

## 지금 당장 가능한 것

### 1) CLI 채점 실행 → JSON 출력 확인

```bash
# golden set 없으면 starter 자동 생성 후 채점까지
python -m eval.judge --golden data/golden_set.jsonl --make-starter --n-starter 10
```

출력 예:
```json
{
  "n": 10,
  "judge": "mock-judge",
  "groundedness_mean": 5.0,
  "relevance_mean": 4.6,
  "actionability_mean": 4.4,
  "overall_mean": 4.67,
  "labeled_supported_frac": 1.0
}
```

- `UPSTAGE_API_KEY`가 있으면 `"judge": "solar-judge"`로 자동 전환 (실제 LLM 채점)
- `labeled_supported_frac`가 1.0이면 **라벨이 전부 True** → calibration 포화 신호. 팀이 label 수정 필요.

### 2) Calibrator 학습 → UI confidence% 개선

채점 후 아래 명령 한 번 실행하면 `data/calibrator.json`이 갱신되고, 다음 Streamlit 실행부터 UI의 신뢰도 배지(trust_badge)가 보정된 confidence를 표시한다.

```bash
python -c "
from eval import iter_golden_set
from trust_gate import Calibrator

items = list(iter_golden_set('data/golden_set.jsonl'))
confs  = [it.response.confidence_raw for it in items]
labels = [1.0 if it.label_supported else 0.0 for it in items]

cal = Calibrator().fit(confs, labels)
cal.save('data/calibrator.json')
print(f'T={cal.temperature:.3f}  항목 {len(items)}건으로 학습 완료')
"
```

> **라벨 다양성이 핵심**: label_supported가 모두 True면 T가 포화되어 confidence%가 과도하게 높아진다. golden_set.jsonl을 텍스트 에디터로 열어 실제로 근거가 부족한 응답의 `"label_supported": false`로 수정한 뒤 재학습하면 정상화된다.

---

## 더 있으면 좋은 것 (우선순위 순)

### 1순위: Streamlit eval 탭 — 라벨링 UI + 채점 차트

현재 golden set 라벨링은 JSONL 파일을 텍스트 에디터로 직접 수정해야 한다. Streamlit에 eval 탭을 추가하면 팀이 브라우저에서 항목을 보면서 라벨을 토글할 수 있다.

구현 방향 (`ui/app.py` 또는 `ui/eval_tab.py`):

```python
# 개념 스케치 — 실제 구현 시 참고
import streamlit as st
from eval import iter_golden_set, save_golden_set, regression_run

with tab_eval:
    items = list(iter_golden_set("data/golden_set.jsonl"))

    # 채점 결과 차트
    result = regression_run(items)
    st.metric("Groundedness", result["groundedness_mean"])
    st.metric("Relevance",    result["relevance_mean"])
    st.metric("Actionability",result["actionability_mean"])
    st.metric("Overall",      result["overall_mean"])
    st.caption(f"label_supported 비율: {result['labeled_supported_frac']:.0%}  |  n={result['n']}")

    # 라벨링 UI
    st.divider()
    st.subheader("라벨 검토")
    updated = False
    for i, item in enumerate(items):
        with st.expander(f"[{item.question_key}] {item.item_id[:8]}... — {item.response.summary[:40]}"):
            st.json(item.response.model_dump(), expanded=False)
            new_label = st.toggle("label_supported", value=item.label_supported, key=f"lbl_{i}")
            if new_label != item.label_supported:
                items[i] = item.model_copy(update={"label_supported": new_label})
                updated = True

    if updated and st.button("저장"):
        save_golden_set(items, "data/golden_set.jsonl")
        st.success("저장 완료. calibrator를 재학습해 주세요.")
```

### 2순위: 자동화 스크립트 — 채점 → calibrator 학습 한 번에

매번 명령을 따로 실행하는 대신 `run_eval.sh` 한 번으로 전 과정을 처리한다.

```bash
# side_project/run_eval.sh
#!/usr/bin/env bash
set -e

GOLDEN="${1:-data/golden_set.jsonl}"
CALIB="data/calibrator.json"

echo "=== Step 1: 채점 ==="
python -m eval.judge --golden "$GOLDEN" --make-starter --n-starter 10

echo ""
echo "=== Step 2: Calibrator 학습 ==="
python -c "
from eval import iter_golden_set
from trust_gate import Calibrator
items  = list(iter_golden_set('$GOLDEN'))
confs  = [it.response.confidence_raw for it in items]
labels = [1.0 if it.label_supported else 0.0 for it in items]
cal    = Calibrator().fit(confs, labels)
cal.save('$CALIB')
print(f'T={cal.temperature:.3f}  항목 {len(items)}건')
"

echo ""
echo "=== 완료: \$CALIB 갱신됨 ==="
echo "Streamlit을 재시작하면 UI trust_badge에 반영됩니다."
```

```bash
# 실행
bash side_project/run_eval.sh
# 또는 특정 golden set 경로 지정
bash side_project/run_eval.sh data/golden_set_v2.jsonl
```

### 3순위: 회귀 알람 — overall_mean이 이전 대비 0.3 이상 떨어지면 경고

프롬프트·모델·Evidence Pack 스키마를 바꿀 때마다 이전 채점 결과와 자동 비교한다.

```python
# side_project/run_eval.sh 에 아래 블록 추가 (Step 2 다음)

echo ""
echo "=== Step 3: 회귀 감지 ==="
python -c "
import json, pathlib, sys

THRESHOLD = 0.3
HISTORY   = pathlib.Path('data/eval_history.jsonl')

# 현재 결과
from eval import iter_golden_set, regression_run
items  = list(iter_golden_set('$GOLDEN'))
result = regression_run(items)
current = result['overall_mean']

# 이전 결과
prev = None
if HISTORY.exists():
    last = HISTORY.read_text().strip().splitlines()
    if last:
        prev = json.loads(last[-1]).get('overall_mean')

# 비교
if prev is not None and (prev - current) >= THRESHOLD:
    print(f'⚠️  회귀 감지: overall_mean {prev:.3f} → {current:.3f}  (차이 {prev-current:.3f} ≥ {THRESHOLD})')
    print('   프롬프트/모델/게이트 변경 이후 품질 저하 가능성을 확인하세요.')
    sys.exit(1)
elif prev is not None:
    print(f'✅  회귀 없음: {prev:.3f} → {current:.3f}')
else:
    print(f'ℹ️  첫 실행. baseline: {current:.3f}')

# 히스토리 기록
with HISTORY.open('a') as f:
    f.write(json.dumps({'overall_mean': current, 'n': result['n'],
                        'groundedness_mean': result['groundedness_mean']}) + '\n')
"
```

히스토리는 `data/eval_history.jsonl`에 누적되어, 실행마다 이전 점수와 비교하고 0.3 이상 떨어지면 exit code 1로 종료한다 (CI에서 실패 처리 가능).
