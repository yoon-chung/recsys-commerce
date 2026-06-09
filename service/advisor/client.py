"""Upstage solar-pro 클라이언트 (OpenAI 호환 엔드포인트 사용).

※ 정확한 base_url / 모델 ID는 Upstage 콘솔 문서로 최종 확인 후 .env에 반영할 것.
   기본값은 형태 예시이며, 다른 OpenAI 호환 LLM(예: OpenAI/Together/Anyscale)으로 바꾸려면
   base_url 과 model 만 교체하면 됨.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .prompts import SYSTEM_PROMPT, build_user_message
from .schema import AdvisorResponse

if TYPE_CHECKING:
    from evidence_pack.schema import EvidencePack


class SolarProClient:
    """OpenAI 호환 클라이언트 — Upstage solar-pro 기본."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai 패키지가 필요합니다: pip install openai>=1.30"
            ) from e

        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "UPSTAGE_API_KEY 환경변수가 없습니다. .env 또는 환경에 키를 설정하거나, "
                "advisor.get_client(force_mock=True)를 사용하세요."
            )
        self.base_url = base_url or os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
        self.model = model or os.getenv("UPSTAGE_MODEL", "solar-pro")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(
        self,
        pack: "EvidencePack",
        item_id: str,
        question_key: str,
        temperature: float = 0.4,
        sample_idx: int = 0,
    ) -> AdvisorResponse:
        """Evidence Pack + 질문 → 검증된 AdvisorResponse.

        Args:
            sample_idx: SelfCheckGPT용 샘플 인덱스. >0 이면 temperature를 살짝 올려 다양성 확보.
        """
        user_msg = build_user_message(pack, item_id, question_key)
        # SelfCheck 샘플링: 같은 프롬프트로 temperature만 올려 N번 호출
        temp = temperature if sample_idx == 0 else max(temperature, 0.7)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=temp,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        if not raw:
            raise RuntimeError("LLM 응답이 비어있습니다")
        # pydantic이 스키마 위반을 잡아준다 (hard_gate의 1차 방어선)
        return AdvisorResponse.model_validate_json(raw)

    def generate_samples(
        self,
        pack: "EvidencePack",
        item_id: str,
        question_key: str,
        n: int = 3,
        temperature: float = 0.7,
    ) -> list[AdvisorResponse]:
        """SelfCheckGPT용 N개 샘플."""
        return [
            self.generate(pack, item_id, question_key, temperature=temperature, sample_idx=i)
            for i in range(n)
        ]


if __name__ == "__main__":
    # API 키가 있을 때만 가벼운 호출 테스트
    if not os.getenv("UPSTAGE_API_KEY"):
        print("UPSTAGE_API_KEY가 없어 실제 호출은 생략. mock_client.py를 사용하세요.")
    else:
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
        from evidence_pack import MockAdapter, build_evidence_pack, iter_evidence_pack

        mock = MockAdapter(n_users=2, top_k=5)
        build_evidence_pack(mock, mock, "data/_advisor_test.jsonl", max_users=1)
        pack = next(iter_evidence_pack("data/_advisor_test.jsonl"))
        c = SolarProClient()
        resp = c.generate(pack, pack.recommendations[0].item_id, "why")
        print(resp.model_dump_json(indent=2))
