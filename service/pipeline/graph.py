"""graph.py — LangGraph 조립 + compile.

사용 예:
    from pipeline.graph import build_app
    app, resources = build_app(data_dir="data")
    result = app.invoke({
        "message": "user_00001 추천 이유 알려줘",
        "question_key": "why",
    })
    print(result["response_text"])
    print(result["trust_badge"])
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .graph_state import GraphState
from .nodes import (
    ResourceBundle,
    make_node_alias_resolver,
    make_node_general_chat,
    make_node_hard_gate,
    make_node_intent_router,
    make_node_pack_loader,
    make_node_profile_loader,
    make_node_solar_explainer,
    route_after_alias,
    route_after_intent,
)


def build_app(
    data_dir: str | Path = "data",
    evidence_pack_path: str | Path | None = None,
    checkpointer: MemorySaver | None = None,
):
    """LangGraph app + ResourceBundle 반환.

    checkpointer=None: stateless (단발 응답).
    checkpointer=MemorySaver(): thread_id 별 multi-turn 대화 누적.
    """
    resources = ResourceBundle.load(data_dir=data_dir, evidence_pack_path=evidence_pack_path)

    graph = StateGraph(GraphState)
    graph.add_node("intent_router",   make_node_intent_router(resources))
    graph.add_node("alias_resolver",  make_node_alias_resolver(resources))
    graph.add_node("profile_loader",  make_node_profile_loader(resources))
    graph.add_node("pack_loader",     make_node_pack_loader(resources))
    graph.add_node("solar_explainer", make_node_solar_explainer(resources))
    graph.add_node("hard_gate",       make_node_hard_gate(resources))
    graph.add_node("general_chat",    make_node_general_chat(resources))

    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges(
        "intent_router", route_after_intent,
        {"general_chat": "general_chat", "alias_resolver": "alias_resolver"},
    )
    graph.add_conditional_edges(
        "alias_resolver", route_after_alias,
        {"end": END, "profile_loader": "profile_loader"},
    )
    graph.add_edge("profile_loader",  "pack_loader")
    graph.add_edge("pack_loader",     "solar_explainer")
    graph.add_edge("solar_explainer", "hard_gate")
    graph.add_edge("hard_gate",       END)
    graph.add_edge("general_chat",    END)

    app = graph.compile(checkpointer=checkpointer)
    return app, resources