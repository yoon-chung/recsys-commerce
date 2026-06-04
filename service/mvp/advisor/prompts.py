"""프롬프트 — System + 빠른질문 + user message 빌더 + 의도 분류 상수.

설계 원칙:
- LLM은 추천을 만들지 않는다. Evidence Pack 안의 사실만으로 답한다. (shopping 모드)
- 의도 분류로 general / shopping 경로를 분리한다.
- item_alias / user_alias는 표시 전용. LLM이 alias 문자열에서 속성을 추론하면 안 된다.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evidence_pack.schema import EvidencePack


# ─── 의도 분류 ───────────────────────────────────────────────────────────────

# 키워드 매칭 우선 (Solar Pro API 호출 없음)
SHOPPING_KEYWORDS: frozenset[str] = frozenset({
    "추천", "상품", "쇼핑", "구매", "브랜드", "뭐 살까", "골라줘",
    "어울리는", "왜 추천", "살까", "더 싸", "더 싼", "저렴", "가격",
    "취향", "신상", "다시", "재방문", "대안",
})

INTENT_CLASSIFY_SYSTEM = """사용자 메시지의 의도를 다음 세 가지 중 하나로 분류하세요.

- shopping : 상품 추천, 구매 상담, 브랜드/카테고리 탐색 등 쇼핑 관련
- user_id  : 유저 ID(user_00001 형식) 입력, 또는 본인 추천 조회가 주목적인 메시지
- general  : 그 외 일반 대화

반드시 ["shopping", "user_id", "general"] 중 하나만 출력하세요. 다른 텍스트 금지.""".strip()


def classify_intent(message: str) -> str:
    """키워드 우선 의도 분류. Solar Pro fallback은 호출자가 처리."""
    import re
    if re.search(r"\buser_\d+\b", message):
        return "user_id"
    if any(kw in message for kw in SHOPPING_KEYWORDS):
        return "shopping"
    return "unknown"  # caller가 Solar Pro로 3-class 분류


# ─── 빠른질문 (UI 노출 키) ───────────────────────────────────────────────────

# 사용자 탭 + 운영자 탭의 정해진 질문 — shopping 모드에서만 사용
QUICK_QUESTIONS: dict[str, str] = {
    # 사용자 탭
    "why": "이 상품이 왜 저에게 추천되었나요?",
    "should_buy": "지금 살 만한가요?",
    "cheaper": "더 저렴한 대안이 있나요?",
    "fit_preference": "제 취향과 왜 맞나요?",
    "revisit": "왜 이 상품을 다시 보면 좋을까요?",
    # 운영자 탭
    "category_trend": "이 카테고리/브랜드가 왜 뜨고 있나요?",
    "promotion_candidate": "프로모션 후보로 적합한가요?",
    "discontinue_candidate": "단종 후보로 검토할 만한가요?",
}


# ─── Shopping 모드 System Prompt (Evidence Pack 바인딩) ──────────────────────

SYSTEM_PROMPT = """당신은 커머스 추천에 대한 어드바이저입니다. 다음 규칙을 반드시 지키세요.

규칙:
1. 추천을 새로 만들지 마세요. 아래 사용자가 제공한 Evidence Pack 안의 사실만으로 답하세요.
2. Evidence Pack에 없는 내용은 절대 만들지 마세요. 근거가 없으면 그렇다고 답하세요.
3. 모든 주장(claim)에는 그 근거가 되는 evidence_ref 키를 정확히 인용하세요.
   허용된 키는 사용자 메시지에 화이트리스트로 제공됩니다. 그 외 키는 사용 금지.
4. 응답은 반드시 다음 JSON 스키마를 따르세요 — 다른 어떤 텍스트도 출력 금지.
5. summary는 한국어 1~2문장, claims는 2~4개, 각 claim의 text는 짧고 구체적으로.
6. confidence_raw는 0.0~1.0 사이로, 본인이 얼마나 확신하는지 정직하게.
7. brand/category_code/category_l2는 데이터셋 내부 라벨입니다. xiaomi/apple/samsung 같은 이름도
   외부 상식으로 전자제품이라고 해석하지 말고, Evidence Pack에 있는 카테고리·브랜드 라벨 그대로만 말하세요.
8. item_alias / user_alias는 표시용 별칭입니다. alias 문자열(예: shoes.keds.kapika.mid_0001)에서
   상품 특성을 추론하지 마세요. 속성은 Evidence Pack의 구조화 필드만 사용하세요.
9. 재방문(revisit) 질문은 signals.revisit_score, signals.last_event_type,
   signals.last_interaction_days, signals.revisit_event_count,
   signals.cart_without_purchase 같은 구조화 근거가 있을 때만 설명하세요.

응답 JSON 스키마:
{
  "user_id": "<원본>",
  "item_id": "<원본>",
  "question_key": "<원본>",
  "summary": "1-2문장 한국어 요약",
  "claims": [
    {"text": "짧은 주장", "evidence_ref": "허용된 키 중 하나"}
  ],
  "decision_hint": "이 사용자에게 줄 1문장 결정 조언",
  "confidence_raw": 0.0
}
"""


def build_user_message(
    pack: "EvidencePack",
    item_id: str,
    question_key: str,
) -> str:
    """LLM에 보낼 user 메시지 — Evidence Pack + 화이트리스트 + 질문."""
    q_text = QUICK_QUESTIONS.get(question_key, question_key)
    evidence_keys = sorted(pack.evidence_keys())
    # alias는 표시용 — evidence_ref 화이트리스트에서 제외됨
    alias_hint = ""
    if pack.user_alias:
        alias_hint = f"사용자 표시 별칭 (참고용 only): {pack.user_alias}\n"
    pack_json = pack.model_dump_json()
    return (
        f"사용자 ID: {pack.user_id}\n"
        f"{alias_hint}"
        f"대상 아이템: {item_id}\n"
        f"질문 키: {question_key}\n"
        f"질문: {q_text}\n\n"
        f"허용된 evidence_ref 키 (이 중에서만 인용 가능):\n"
        f"{json.dumps(evidence_keys, ensure_ascii=False, indent=2)}\n\n"
        f"Evidence Pack:\n{pack_json}\n"
    )
