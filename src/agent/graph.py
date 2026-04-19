from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .nodes import (
    call_llm_node,
    guard_tool_call_node,
    tool_node,
    should_excute_tool,
    should_execute_tool_after_guard,
)

class NoteTakingAgent:
    def __init__(self):
        self.state_graph = StateGraph(AgentState)
        self.agent = self._build_graph()
    
    def _build_graph(self):
        self.state_graph.add_node("call_llm", call_llm_node)
        self.state_graph.add_node("guard_tool_call", guard_tool_call_node)
        self.state_graph.add_node("execute_tool", tool_node)

        # Standard LangGraph entry point
        self.state_graph.add_edge(START, "call_llm")

        self.state_graph.add_conditional_edges(
            "call_llm", 
            should_excute_tool, 
            {"tools": "guard_tool_call", "end": END}
        )

        self.state_graph.add_conditional_edges(
            "guard_tool_call",
            should_execute_tool_after_guard,
            {"execute_tool": "execute_tool", "retry_llm": "call_llm"}
        )

        self.state_graph.add_edge("execute_tool", "call_llm")

        return self.state_graph.compile()
    
    def invoke(self, state: AgentState):
        return self.agent.invoke(state)


if __name__ == "__main__":
    import os

    # Ensure the docs directory exists
    os.makedirs("docs", exist_ok=True)

    agent = NoteTakingAgent("test")
    graph_png_bytes = agent.agent.get_graph().draw_mermaid_png()
    
    # Write the raw bytes to a PNG file
    with open("docs/agent_graph.png", "wb") as f:
        f.write(graph_png_bytes)
    
    print("Graph successfully rendered to docs/agent_graph.png")