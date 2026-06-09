"""nodes.py — LangGraph 노드 함수 모음.

각 노드는 GraphState dict 일부를 받아 일부를 업데이트한다.
의존 자원(catalog, recs, recency, evidence_pack)은 ResourceBundle에 묶어 전역으로 보유.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from advisor import get_client
from advisor.prompts import classify_intent
from advisor.schema import AdvisorResponse
from evidence_pack.schema import EvidencePack
from trust_gate.hard_gate import check_response

from .graph_state import GraphState
from .orchestrator import recommend_all_sections


# ── 리소스 번들 (한 번 로드, 노드들이 공유) ──────────────────────────────────

@dataclass
class ResourceBundle:
    """런타임에 1회 로드되는 정적 자원."""
    catalog: dict[str, dict]
    recs: dict[str, list[str]]
    recency_pool: list[str]
    pack_index: dict[str, EvidencePack]   # user_alias → EvidencePack
    profile_db_path: Path
    cf_neighbors: dict[str, list[str]] | None = None  # user_alias → list[neighbor_alias] (FAISS 빌드 시)

    @classmethod
    def load(
        cls,
        data_dir: str | Path = "data",
        evidence_pack_path: str | Path | None = None,
    ) -> "ResourceBundle":
        import pickle

        data_dir = Path(data_dir)
        evidence_pack_path = Path(evidence_pack_path or data_dir / "evidence_pack.jsonl")

        with (data_dir / "item_catalog.json").open(encoding="utf-8") as f:
            catalog = json.load(f)
        with (data_dir / "user_recommendations.json").open(encoding="utf-8") as f:
            recs = json.load(f)
        with (data_dir / "recency_pool.json").open(encoding="utf-8") as f:
            recency_pool = json.load(f)

        pack_index: dict[str, EvidencePack] = {}
        with evidence_pack_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                pack = EvidencePack.model_validate_json(line)
                if pack.user_alias:
                    pack_index[pack.user_alias] = pack

        # CF 이웃 (선택) — FAISS 미빌드 시 None → collaborative 섹션 빈 채로 노출 안 됨
        cf_neighbors = None
        neighbors_path = data_dir / "user_neighbors.npy"
        alias_to_row_path = data_dir / "user_alias_to_row.pkl"
        row_to_alias_path = data_dir / "row_to_user_alias.pkl"
        if neighbors_path.exists() and alias_to_row_path.exists() and row_to_alias_path.exists():
            import numpy as np
            neighbors = np.load(neighbors_path)
            with alias_to_row_path.open("rb") as f:
                alias_to_row = pickle.load(f)
            with row_to_alias_path.open("rb") as f:
                row_to_alias = pickle.load(f)
            # FAISS는 active user < k+1일 때 빈 슬롯을 -1로 채움 → 필터링
            cf_neighbors = {
                alias: [row_to_alias[int(r)] for r in neighbors[row] if r >= 0]
                for alias, row in alias_to_row.items()
            }

        return cls(
            catalog=catalog,
            recs=recs,
            recency_pool=recency_pool,
            pack_index=pack_index,
            profile_db_path=data_dir / "user_profiles.db",
            cf_neighbors=cf_neighbors,
        )


# ── 노드: intent_router ──────────────────────────────────────────────────────

USER_ALIAS_PATTERN = re.compile(r"\buser_\d+\b")


def make_node_intent_router(_resources: ResourceBundle):
    # 매 턴 stale 출력 필드 초기화 (MemorySaver multi-turn 대비)
    _STALE_RESET = {
        "response_text": None,
        "trust_badge": None,
        "hard_gate_errors": None,
        "advisor_response": None,
        "pack_json": None,
        "sections": None,
        "intent": None,
    }

    def node(state: GraphState) -> GraphState:
        msg = state.get("message", "")
        if USER_ALIAS_PATTERN.search(msg):
            return {**_STALE_RESET, "intent": "user_id"}
        kw_intent = classify_intent(msg)
        if kw_intent in ("shopping", "user_id"):
            return {**_STALE_RESET, "intent": kw_intent}
        return {**_STALE_RESET, "intent": "general"}
    return node


# ── 노드: alias_resolver ─────────────────────────────────────────────────────

def make_node_alias_resolver(_resources: ResourceBundle):
    def node(state: GraphState) -> GraphState:
        msg = state.get("message", "")
        m = USER_ALIAS_PATTERN.search(msg)
        if m:
            return {"user_alias": m.group(0)}
        if state.get("user_alias"):
            return {}
        return {
            "response_text": "추천을 위해 user_00001 형식의 ID가 필요합니다.",
            "trust_badge": "unverified",
        }
    return node


# ── 노드: profile_loader (orchestrator로 5섹션 생성) ──────────────────────────

def make_node_profile_loader(resources: ResourceBundle):
    def node(state: GraphState) -> GraphState:
        ua = state.get("user_alias")
        if not ua:
            return {"response_text": "유저 별칭이 없습니다.", "trust_badge": "unverified"}
        sections = recommend_all_sections(
            ua, resources.recs, resources.catalog, resources.recency_pool,
            resources.profile_db_path, top_per_section=5,
            cf_neighbors=resources.cf_neighbors,
        )
        return {"sections": sections}
    return node


# ── 노드: pack_loader ────────────────────────────────────────────────────────

def make_node_pack_loader(resources: ResourceBundle):
    def node(state: GraphState) -> GraphState:
        ua = state.get("user_alias")
        if not ua:
            return {}
        pack = resources.pack_index.get(ua)
        if pack is None:
            return {"response_text": f"{ua}의 Evidence Pack이 없습니다 (sample 빌드 외부).",
                    "trust_badge": "unverified"}
        # selected_item_id 기본값: submission top-1
        sel = state.get("selected_item_id")
        if not sel and pack.recommendations:
            sel = pack.recommendations[0].item_id
        return {"pack_json": pack.model_dump_json(), "selected_item_id": sel}
    return node


# ── 노드: solar_explainer ────────────────────────────────────────────────────

def make_node_solar_explainer(_resources: ResourceBundle):
    def node(state: GraphState) -> GraphState:
        pack_json = state.get("pack_json")
        item_id = state.get("selected_item_id")
        qkey = state.get("question_key", "why")
        if not (pack_json and item_id):
            return {}
        pack = EvidencePack.model_validate_json(pack_json)
        client = get_client()  # API 없으면 자동 mock
        try:
            resp = client.generate(pack, item_id, qkey)
            return {"advisor_response": resp.model_dump()}
        except Exception as e:
            return {"response_text": f"LLM 호출 실패: {e}", "trust_badge": "rejected"}
    return node


# ── 노드: hard_gate_check ────────────────────────────────────────────────────

def make_node_hard_gate(_resources: ResourceBundle):
    def node(state: GraphState) -> GraphState:
        pack_json = state.get("pack_json")
        resp_dict = state.get("advisor_response")
        if not (pack_json and resp_dict):
            return {}
        pack = EvidencePack.model_validate_json(pack_json)
        resp = AdvisorResponse.model_validate(resp_dict)
        errors = check_response(pack, resp)
        if errors:
            return {
                "hard_gate_errors": errors,
                "trust_badge": "rejected",
                "response_text": f"[검증 실패] {resp.summary}\n오류: {'; '.join(errors[:2])}",
            }
        return {
            "hard_gate_errors": [],
            "trust_badge": "verified",
            "response_text": resp.summary,
        }
    return node


# ── 노드: general_chat ───────────────────────────────────────────────────────

def make_node_general_chat(_resources: ResourceBundle):
    def node(state: GraphState) -> GraphState:
        msg = state.get("message", "")
        return {
            "response_text": (
                "⚠️ 일반 대화 모드입니다. Evidence Pack 기반 검증이 적용되지 않습니다.\n"
                f"입력: {msg}"
            ),
            "trust_badge": "unverified",
        }
    return node


# ── routing helpers ──────────────────────────────────────────────────────────

def route_after_intent(state: GraphState) -> str:
    intent = state.get("intent")
    if intent == "general":
        return "general_chat"
    return "alias_resolver"


def route_after_alias(state: GraphState) -> str:
    if state.get("response_text"):
        return "end"
    return "profile_loader"