
import json
import sys
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, START, END
from src.tools.prediction_tool import predict_failure


# Shared information passed between graph nodes
class AgentState(TypedDict, total=False):
    pipeline_name: str
    input_features: dict
    prediction: dict
    route_taken: str
    final_message: str


# Node 1: Validate the request
def validate_input(state: AgentState) -> dict:
    name = state.get("pipeline_name")
    features = state.get("input_features")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Pipeline name must be provided")

    if not isinstance(features, dict):
        raise ValueError("Input features must be a dictionary")

    return {"pipeline_name": name.strip()}


# Node 2: Call the ML prediction tool
def prediction_node(state: AgentState) -> dict:
    prediction = predict_failure(**state["input_features"])
    return {"prediction": prediction}


# Conditional edge: Decide which node runs next
def route_by_risk(state: AgentState) -> str:
    if state["prediction"]["predicted_failure"]:
        return "risk_review"
    return "normal_processing"


# Node 3A: Handle elevated risk
def risk_review(state: AgentState) -> dict:
    probability = state["prediction"]["failure_probability"]

    return {
        "route_taken": "risk_review",
        "final_message": (
            f"{state['pipeline_name']} has elevated predicted "
            f"failure risk ({probability:.2%}). "
            "Recommend further investigation."
        ),
    }


# Node 3B: Handle lower risk
def normal_processing(state: AgentState) -> dict:
    probability = state["prediction"]["failure_probability"]

    return {
        "route_taken": "normal_processing",
        "final_message": (
            f"{state['pipeline_name']} has lower predicted "
            f"failure risk ({probability:.2%}). "
            "Continue standard operational checks."
        ),
    }


# Build and compile the graph
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("validate_input", validate_input)
    workflow.add_node("prediction", prediction_node)
    workflow.add_node("risk_review", risk_review)
    workflow.add_node("normal_processing", normal_processing)

    workflow.add_edge(START, "validate_input")
    workflow.add_edge("validate_input", "prediction")

    workflow.add_conditional_edges(
        "prediction",
        route_by_risk,
        {
            "risk_review": "risk_review",
            "normal_processing": "normal_processing",
        },
    )

    workflow.add_edge("risk_review", END)
    workflow.add_edge("normal_processing", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()

    example = {
        "pipeline_name": "payment_etl",
        "input_features": {
            "expected_workload": 9000,
            "cpu_usage_pct": 92.0,
            "memory_usage_pct": 88.0,
            "recent_failure_rate": 0.20,
        },
    }

    result = graph.invoke(example)
    print(json.dumps(result, indent=2))

    diagram = graph.get_graph().draw_mermaid()

    diagram_path = ROOT / "docs" / "langgraph_workflow.md"
    diagram_path.write_text(
        "# AgentOps LangGraph Workflow\n\n"
        "```mermaid\n"
        + diagram
        + "\n```\n",
        encoding="utf-8",
    )

    print(f"\nGraph documentation saved to: {diagram_path}")
