import json
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from typing import List

from .state import AgentState
from ..llm.provider import llm
from ..tools.note_tools import tools
from .prompts import get_system_prompt, get_compaction_prompt, get_guard_prompt

logger = logging.getLogger(__name__)

DESTRUCTIVE_TOOLS = {"delete_note", "update_note"}

MAX_HISTORY_MESSAGES = 20

def _retrieve_or_compact_history(messages: List[BaseMessage], active_note_ids: List[str], model) -> List[BaseMessage]:
    """Returns the conversation history or compact the conversation when number of messages is equal to the `MAXMAX_HISTORY_MESSAGES`"""
    conversational_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(conversational_messages) == MAX_HISTORY_MESSAGES:
        compaction_prompt = get_compaction_prompt()
        conversational_messages = [compaction_prompt] + conversational_messages
        resp = model.invoke(conversational_messages)
        conversational_messages = [resp]

    return [get_system_prompt(active_note_ids)] + conversational_messages


def call_llm_node(state: AgentState) -> AgentState:
    model = llm.client.bind_tools(tools)

    messages = state.get("messages", [])
    active_note_ids = state.get("active_note_ids", [])
    pending_confirmation = state.get("pending_confirmation")
    error = state.get("error", "")

    # Extract newly seen Note IDs from tool responses
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            break
        try:
            data = json.loads(m.content)
            if isinstance(data, dict) and "id" in data:
                if data["id"] not in active_note_ids:
                    active_note_ids.append(data["id"])
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "id" in item:
                        if item["id"] not in active_note_ids:
                            active_note_ids.append(item["id"])
        except Exception:
            pass

    messages_to_pass = _retrieve_or_compact_history(
        messages=messages, 
        active_note_ids=active_note_ids,
        model=model
    )

    # Re-prompt the LLM if the guard returned an error on the previous turn
    if error:
        logger.debug("LLM re-prompting with error: %s", error)
        messages_to_pass.append(HumanMessage(content=f"Previous error: {error}. Please try again."))

    for m in messages_to_pass:
        if isinstance(m, ToolMessage) and not m.content:
            m.content = "No results found."

    response = model.invoke(messages_to_pass)

    return {
        "messages": [response],
        "error": "",
        "active_note_ids": active_note_ids,
        "pending_confirmation": pending_confirmation,
    }


def guard_tool_call_node(state: AgentState) -> AgentState:
    """
    Safety guard applied before every tool execution. Enforces:
    1. ID validation — tools that require an ID must receive one from active_note_ids
       (prevents hallucinated IDs).
    2. Confirmation gate — destructive tools (delete/update) require the user to
       have explicitly confirmed before execution proceeds.
    """
    messages = state.get("messages", [])
    active_note_ids = state.get("active_note_ids", [])
    pending_confirmation = state.get("pending_confirmation")

    if not messages or not isinstance(messages[-1], AIMessage) or not messages[-1].tool_calls:
        # Nothing to guard
        return {
            "active_note_ids": active_note_ids,
            "pending_confirmation": pending_confirmation,
            "error": "",
        }

    for tool_call in messages[-1].tool_calls:
        tool_name = tool_call.get("name")
        args = tool_call.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        note_id = str(args.get("id", "")).strip()

        # --- ID validation ---
        if tool_name in ("get_note_by_id", "update_note", "delete_note"):
            if not note_id:
                return {
                    "error": f"Tool '{tool_name}' requires an 'id'. Search for the note first.",
                    "pending_confirmation": None,
                    "active_note_ids": active_note_ids,
                }
            if note_id not in active_note_ids:
                return {
                    "error": (
                        f"CRITICAL: Note ID '{note_id}' is not in your context memory. "
                        f"Do NOT retry this exact call. You MUST call 'search_notes' or "
                        f"'semantic_search' first to obtain the correct ID."
                    ),
                    "pending_confirmation": None,
                    "active_note_ids": active_note_ids,
                }

        # --- Confirmation gate for destructive tools ---
        if tool_name not in DESTRUCTIVE_TOOLS:
            continue

        # Use an LLM call to verify if the user explicitly confirmed this action
        recent_history = "\n".join([f"{m.type}: {m.content}" for m in messages[-4:] if m.content])
        verification_prompt = get_guard_prompt(tool_name, recent_history)
        response = llm.client.invoke([HumanMessage(content=verification_prompt)])
        
        if "YES" in response.content.upper():
            # Confirmed - clear pending and allow through
            return {
                "pending_confirmation": None,
                "active_note_ids": active_note_ids,
                "error": "",
            }
        else:
            # Block it and ask for confirmation
            return {
                "error": (
                    f"Action '{tool_name}' blocked. The user has not confirmed yet! "
                    f"You MUST ask the user for explicit confirmation before proceeding. "
                    f"Say: 'I am about to {tool_name}. Are you sure you want to proceed?' and wait for their reply. "
                    f"DO NOT call the tool again until they confirm."
                ),
                "pending_confirmation": {"tool": tool_name, "args": args},
                "active_note_ids": active_note_ids,
            }

    # All checks passed
    return {
        "active_note_ids": active_note_ids,
        "pending_confirmation": pending_confirmation,
        "error": "",
    }


tool_node = ToolNode(tools)


def should_excute_tool(state: AgentState) -> str:
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage):
        if messages[-1].tool_calls:
            return "tools"
    return "end"


def should_execute_tool_after_guard(state: AgentState) -> str:
    if state.get("error"):
        return "retry_llm"
    return "execute_tool"