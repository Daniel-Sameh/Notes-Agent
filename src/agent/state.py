from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    """
    The Agent's State for the conversational note-taking agent.
    Includes context memory, disambiguation states, and error handling.
    """
    # Core multi-turn conversation tracking
    messages: Annotated[List[AnyMessage], add_messages]
    
    # To handle destructive action confirmation (update/delete)
    # Holds details of the pending action structured as a dict (e.g. {"tool": "delete_note", "args": {"id": 5}})
    pending_confirmation: Optional[Dict[str, Any]]
    
    # To handle intent disambiguation and object reference
    # I keep track of the most recently discussed note IDs
    active_note_ids: List[int]
    
    error: Optional[str]
    
    user_id: str