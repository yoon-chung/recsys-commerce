"""LLM-as-Judge — G-Eval 식 다차원 채점.

원 논문:
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023)
- Liu et al., "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" (EMNLP 2023)

차원: groundedness / relevance / actionability (1~5).
런타임이 아닌 **오프라인 회귀 지표**. 프롬프트/모델/게이트를 바꿨을 때 품질 회귀를 자동 감지.

Bias 완화:
- Position bias — 이 코드는 absolute scoring이라 영향 작음. pairwise 비교 시 순서 swap 필요.
- Verbosity bias — 채점관에 길이가 점수에 영향 주지 말라고 명시.
- Self-enhancement bias — judge LLM과 advisor LLM이 같으면 점수가 부풀려질 수 있음. 가능하면 다른 모델 사용.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from evidence_pack.schema import EvidencePack
    from advisor.schema import AdvisorResponse
    from .golden_set import GoldenItem


JUDGE_SYSTEM_PROMPT = """당신은 LLM 응답 채점관입니다. 주어진 Evidence Pack과 LLM 응답을 보고
세 가지 차원을 1~5로 평가하고, 각각 짧은 이유를 답하세요.

차원:
- groundedness (1-5): 응답의 모든 주장이 Evidence Pack의 신호로 실제로 뒷받침되는가. 일부라도 근거 밖이면 감점.
- relevance (1-5): 질문(question_key)에 정확히 답했는가.
- actionability (1-5): 사용자/운영자가 다음 행동을 결정하는 데 도움이 되는가.

채점 규칙:
- 길이로 점수 주지 말 것. 짧고 정확하면 5, 길고 산만하면 감점.
- 모든 차원은 정수 1~5.
- overall은 세 점수의 평균 (소수 1자리).

응답 JSON:
{
  "groundedness": 0,
  "groundedness_reason": "...",
  "relevance": 0,
  "relevance_reason": "...",
  "actionability": 0,
  "actionability_reason": "...",
  "overall": 0.0
}
"""


class JudgeScore(BaseModel):
    groundedness: int = Field(ge=1, le=5)
    groundedness_reason: str = ""
    relevance: int = Field(ge=1, le=5)
    relevance_reason: str = ""
    actionability: int = Field(ge=1, le=5)
    actionability_reason: str = ""
    overall: float = Field(ge=1.0, le=5.0)


# ---------- MockJudge -----------------------------------------------------


class MockJudge:
    """결정적 mock judge — hard_gate / self_check / claim 구조로 점수 산정."""

    name = "mock-judge"

    def score(self, pack: "EvidencePack", response: "AdvisorResponse") -> JudgeScore:
        # groundedness: hard_gate 통과 + wobbly flag 없음이 만점
        g = 5 if response.hard_gate_passed else 1
        for f in response.flags:
            if "wobbly" in f.lower():
                g -= 1
        g = max(1, min(5, g))

        # relevance: claims가 2개 이상 + summary 존재
        r = 3
        if len(response.claims) >= 2 and response.summary:
            r = 4
        if len(response.claims) >= 3 and response.summary:
            r = 5
        if len(response.claims) == 0:
            r = 1
        r = max(1, min(5, r))

        # actionability: decision_hint 존재
        a = 4 if response.decision_hint else 2
        if response.confidence_calibrated is not None and response.confidence_calibrated > 0.5:
            a = min(5, a + 1)
        a = max(1, min(5, a))

        return JudgeScore(
            groundedness=g,
            groundedness_reason="hard_gate_passed + wobbly flag 없음 기준",
            relevance=r,
            relevance_reason=f"claims {len(response.claims)}개, summary 존재 여부 기준",
            actionability=a,
            actionability_reason="decision_hint + 보정 confidence 기준",
            overall=round((g + r + a) / 3, 1),
        )


# ---------- LLMJudge ------------------------------------------------------


class LLMJudge:
    """실제 LLM judge — OpenAI 호환 엔드포인트 사용 (기본: solar-pro)."""

    name = "solar-judge"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError("openai 패키지 필요") from e
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError("UPSTAGE_API_KEY 없음")
        self.base_url = base_url or os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
        self.model = model or os.getenv("UPSTAGE_MODEL", "solar-pro")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def score(self, pack: "EvidencePack", response: "AdvisorResponse") -> JudgeScore:
        user_msg = (
            f"질문 키: {response.question_key}\n\n"
            f"Evidence Pack:\n{pack.model_dump_json()}\n\n"
            f"LLM 응답:\n{response.model_dump_json()}\n"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        return JudgeScore.model_validate_json(raw)


def get_judge(force_mock: bool | None = None):
    """환경에 따라 실제/mock judge 자동 선택."""
    if force_mock is None:
        force_mock = os.getenv("ADVISOR_FORCE_MOCK") == "1" or not os.getenv("UPSTAGE_API_KEY")
    if force_mock:
        return MockJudge()
    return LLMJudge()


# ---------- 회귀 채점 -----------------------------------------------------


def regression_run(
    items: list["GoldenItem"],
    judge: "MockJudge | LLMJudge | None" = None,
) -> dict:
    """golden set 전체에 judge를 돌려 집계 점수 반환."""
    judge = judge or get_judge()
    rows = []
    for it in items:
        sc = judge.score(it.pack, it.response)
        rows.append({
            "user_id": it.pack.user_id,
            "item_id": it.item_id,
            "question_key": it.question_key,
            "label_supported": it.label_supported,
            "groundedness": sc.groundedness,
            "relevance": sc.relevance,
            "actionability": sc.actionability,
            "overall": sc.overall,
        })
    if not rows:
        return {"n": 0, "rows": []}

    def _mean(key):
        return round(statistics.mean(r[key] for r in rows), 3)

    return {
        "n": len(rows),
        "judge": judge.name,
        "groundedness_mean": _mean("groundedness"),
        "relevance_mean": _mean("relevance"),
        "actionability_mean": _mean("actionability"),
        "overall_mean": _mean("overall"),
        "labeled_supported_frac": round(sum(1 for r in rows if r["label_supported"]) / len(rows), 3),
        "rows": rows,
    }


if __name__ == "__main__":
    import argparse
    from .golden_set import iter_golden_set, make_starter_golden_set
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="data/golden_set.jsonl")
    parser.add_argument("--make-starter", action="store_true", help="없으면 starter 생성")
    parser.add_argument("--n-starter", type=int, default=10)
    args = parser.parse_args()

    if args.make_starter and not __import__("pathlib").Path(args.golden).exists():
        make_starter_golden_set(args.n_starter, args.golden)

    items = list(iter_golden_set(args.golden))
    if not items:
        print(f"ERROR: golden set이 비어있습니다: {args.golden}. --make-starter 옵션 사용.")
        raise SystemExit(1)
    result = regression_run(items)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, ensure_ascii=False))
    print(f"\n총 {result['n']}건 채점 완료. 상세는 rows 키 참고.")
