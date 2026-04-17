from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from .state import AgentState
from ..llm.provider import llm
from ..tools.note_tools import tools
from .prompts import get_system_prompt

def call_llm_node(state: AgentState) -> AgentState:
    model = llm.client.bind_tools(tools)
    
    messages = state.get("messages", [])
    active_note_ids = state.get("active_note_ids", [])
    error = state.get("error", "")
    
    # Start with the updated SystemMessage
    messages_to_pass = [get_system_prompt(active_note_ids)]
    
    # Append all actual conversational history (excluding any bare SystemMessage)
    messages_to_pass.extend([m for m in messages if not isinstance(m, SystemMessage)])
    
    # If there was a previous tool error, append it as a HumanMessage so the LLM fixes its mistake
    if error:
        messages_to_pass.append(HumanMessage(content=f"Previous error: {error}. Please try again."))
    
    # Invoke the model with our newly constructed, safe array
    response = model.invoke(messages_to_pass)
    
    return {
        "messages": [response],
        "error": ""
    }

tool_node = ToolNode(tools)

def should_excute_tool(state: AgentState) -> str:
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage):
        if messages[-1].tool_calls:
            return "tools"
        
    return "end"