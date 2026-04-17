from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .nodes import call_llm_node, tool_node, should_excute_tool

class NoteTakingAgent:
    def __init__(self, user_id: str):
        self.state_graph = StateGraph(AgentState)
        self.agent = self._build_graph()
    
    def _build_graph(self):
        self.state_graph.add_node("call_llm", call_llm_node)
        self.state_graph.add_node("execute_tool", tool_node)

        # Standard LangGraph entry point
        self.state_graph.add_edge(START, "call_llm")

        self.state_graph.add_conditional_edges(
            "call_llm", 
            should_excute_tool, 
            {"tools": "execute_tool", "end": END}
        )

        self.state_graph.add_edge("execute_tool", "call_llm")

        return self.state_graph.compile()
