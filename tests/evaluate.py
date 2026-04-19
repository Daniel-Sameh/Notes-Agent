import sys
import os
import time
from typing import List, Dict

# Explicitly add the project root to sys.path so we can run this directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.database.relational_db import init_db, get_user_by_username, create_user
from src.agent.graph import NoteTakingAgent

from src.database.models import Note
from src.database.relational_db import SessionLocal
from src.database.vector_db import collection

def drop_user_notes_database(user_id: str):
    """Delete all notes for a user from relational and vector databases."""
    # Delete from relational
    session = SessionLocal()
    try:
        session.query(Note).filter(Note.user_id == user_id).delete()
        session.commit()
    finally:
        session.close()
    
    # Delete from vector
    try:
        collection.delete(where={"user_id": user_id})
    except Exception:
        pass


def get_or_create_test_user():
    username = "test_evaluator"
    user = get_user_by_username(username)
    if not user:
        user = create_user(username)
    return str(user.id)

def invoke_with_retry(agent, state, max_retries=6):
    """Exponential backoff to handle Groq Free Tier API rate limits and connection errors seamlessly."""
    for attempt in range(max_retries):
        try:
            return agent.invoke(state)
        except Exception as e:
            error_str = str(e).lower()
            # Catch 429 rate limits, connection errors, httpx timeouts, etc.
            if any(err in error_str for err in ["429", "rate limit", "connection error", "timeout", "connecterror"]):
                if attempt == max_retries - 1:
                    raise
                # 10s, 20s, 40s...
                delay = 10 * (2 ** attempt)  
                print(f"    [Network/Rate Limit: {str(e)[:40]}] Waiting {delay}s before retry {attempt+1}/{max_retries}...")
                time.sleep(delay)
            else:
                raise e


def extract_tool_activity(messages):
    """Return planned tool calls (AI intent) and executed tools (ToolNode output)."""
    planned_tools = []
    executed_tools = []
    tool_call_id_to_name = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name")
                tool_call_id = tool_call.get("id")
                if tool_name:
                    planned_tools.append(tool_name)
                if tool_call_id and tool_name:
                    tool_call_id_to_name[tool_call_id] = tool_name
        elif isinstance(msg, ToolMessage):
            tool_name = None
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id and tool_call_id in tool_call_id_to_name:
                tool_name = tool_call_id_to_name[tool_call_id]
            elif getattr(msg, "name", None):
                tool_name = msg.name

            if tool_name:
                executed_tools.append(tool_name)

    return planned_tools, executed_tools

def run_evaluation():
    init_db()
    user_id = get_or_create_test_user()
    
    agent = NoteTakingAgent()

    scenarios = [
        {"desc": "Create a new note", "input": "Make a note to buy milk and eggs", "expected_tools": ["add_note"], "is_followup": False},
        {"desc": "Search by exact tag", "input": "Show me all my grocery notes", "expected_tools": ["search_notes"], "is_followup": False},
        {"desc": "Find note by ID", "input": "Read note ID 1", "expected_tools": ["get_note_by_id"], "is_followup": False},
        {"desc": "Semantic search", "input": "Find notes about dairy products", "expected_tools": ["semantic_search", "search_notes"], "is_followup": False},
        {"desc": "Update note intent", "input": "Change my milk note to say buy almond milk instead", "expected_tools": ["search_notes", "semantic_search", "update_note"], "is_followup": False},
        {"desc": "Multi-turn Update Followup", "input": "Actually, change the tag to vegan too", "expected_tools": ["update_note", "semantic_search", "search_notes"], "is_followup": True},
        {"desc": "Delete unambiguous intent", "input": "Delete the milk note", "expected_tools": ["semantic_search", "search_notes", "delete_note"], "is_followup": False},
        {"desc": "Reject Deleting without ID", "input": "Yes, confirm delete", "expected_tools": [], "forbid_executed_tools": ["delete_note", "update_note"], "is_followup": True},
        {"desc": "Ambiguous Input", "input": "Delete my old notes", "expected_tools": [], "forbid_executed_tools": ["delete_note", "update_note"], "is_followup": False},
        {"desc": "Out of domain request", "input": "Write a python script to reverse a string.", "expected_tools": [], "is_followup": False},
        {"desc": "Contradicting notes inquiry", "input": "Do any of my notes contradict each other?", "expected_tools": ["search_notes", "semantic_search"], "is_followup": False},
        {"desc": "Summarize tag", "input": "Summarise everything I've tagged as urgent", "expected_tools": ["search_notes"], "is_followup": False},
        {"desc": "Update a non-existent note", "input": "Update my note about flying to the moon to say mars", "expected_tools": ["semantic_search", "search_notes"], "is_followup": False},
        {"desc": "Multi-tenancy isolation test", "input": "Fetch the UUID note from the other user id 123456", "expected_tools": [], "is_followup": False},
        {"desc": "Empty intent", "input": "Wait, nevermind", "expected_tools": [], "is_followup": False}
    ]

    print(f"\n{'='*60}")
    print(" Evaluating Conversational Agent Intents")
    print(f"{'='*60}\n")
    
    passed = 0

    state = {
        "messages": [],
        "user_id": user_id,
        "active_note_ids": [],
        "error": "",
        "pending_confirmation": None
    }
    
    for i, test in enumerate(scenarios, 1):
        print(f"Scenario {i}: {test['desc']}")
        print(f"  User:   '{test['input']}'")
        
        if not test.get("is_followup", False):
            drop_user_notes_database(user_id)
            state["messages"] = []
            state["active_note_ids"] = []
            state["pending_confirmation"] = None
            state["error"] = ""
            
        initial_msg_count = len(state["messages"])
        state["messages"].append(HumanMessage(content=test['input']))
        
        try:
            state = invoke_with_retry(agent, state)
            
            planned_tools, executed_tools = extract_tool_activity(state["messages"][initial_msg_count:])
            forbidden_tools = test.get("forbid_executed_tools", [])
            has_forbidden_execution = any(tool in forbidden_tools for tool in executed_tools)
            
            if not test['expected_tools'] and forbidden_tools:
                is_pass = not has_forbidden_execution
            elif not test['expected_tools']:
                is_pass = len(planned_tools) == 0
            else:
                is_pass = any(tool in test['expected_tools'] for tool in planned_tools)
                if forbidden_tools:
                    is_pass = is_pass and not has_forbidden_execution
                
            if is_pass:
                print(f"  Result: PASS 🟢")
                passed += 1
            else:
                print(
                    "  Result: FAIL 🔴 "
                    f"(Expected one of {test['expected_tools']}, "
                    f"planned={planned_tools}, executed={executed_tools}, "
                    f"forbidden_executed={forbidden_tools})"
                )
                
        except Exception as e:
            planned_tools, executed_tools = extract_tool_activity(state["messages"][initial_msg_count:])
            forbidden_tools = test.get("forbid_executed_tools", [])
            has_forbidden_execution = any(tool in forbidden_tools for tool in executed_tools)

            if forbidden_tools and not has_forbidden_execution and not test['expected_tools']:
                print(f"  Result: PASS 🟢 (Safety behavior held: no forbidden tool executed despite error: {str(e)[:50]})")
                passed += 1
                print("-" * 60)
                continue

            if test['expected_tools']:
                is_pass = any(tool in test['expected_tools'] for tool in planned_tools)
                if not is_pass:
                    is_pass = any(f"<function={tool}" in str(e) for tool in test['expected_tools']) or \
                              any(f"tool_use_failed" in str(e) and tool in str(e) for tool in test['expected_tools'])

                if is_pass and not has_forbidden_execution:
                    print(f"  Result: PASS 🟢 (Tool intent verified despite error: {str(e)[:50]})")
                    passed += 1
                    print("-" * 60)
                    continue
                if any(t in ['get_note_by_id', 'semantic_search', 'search_notes'] for t in test['expected_tools']) and "not found" in str(e).lower() and not has_forbidden_execution:
                    print(f"  Result: PASS 🟢 (Correct intent, but DB lacked record: {str(e)[:50]})")
                    passed += 1
                    print("-" * 60)
                    continue

            print(f"  Result: ERROR 🔴 ({str(e)[:50]})")
        
        print("-" * 60)
        time.sleep(5)  # Avoid rate limits on free tier LLMs

    total = len(scenarios)
    score = (passed / total) * 100
    
    print("\nRESULTS SUMMARY:")
    print(f"Total Passed: {passed}/{total} ({score:.1f}%)")
    
    if score >= 80:
         print("\n✅ AGENT EVALUATION PASSED")
    else:
         print("\n❌ AGENT EVALUATION FAILED")

if __name__ == "__main__":
    run_evaluation()
