import sys
import os
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.live import Live

# Ensure the database is initialized
from src.database.relational_db import init_db, get_user_by_username, create_user
from src.agent.graph import NoteTakingAgent

console = Console()

def main():
    console.print(Panel.fit("[bold blue]Conversational Note-Taking Agent[/bold blue]", border_style="blue"))
    
    init_db()
    
    username = Prompt.ask("Enter your username to login", default="default_user").strip()
        
    user = get_user_by_username(username)
    if not user:
        console.print(f"[green]Creating new user profile for '{username}'...[/green]")
        user = create_user(username)
    else:
        console.print(f"[green]Welcome back, {username}![/green]")

    user_id = str(user.id)
    
    # Initialize the LangGraph agent
    agent_wrapper = NoteTakingAgent()
    app = agent_wrapper.agent
    
    # Initialize the conversational state
    state = {
        "messages": [],
        "user_id": user_id,
        "active_note_ids": [],
        "error": "",
        "pending_confirmation": None
    }
    
    console.print("[dim]Type 'exit' or 'quit' to exit.[/dim]\n")
    
    while True:
        user_input = Prompt.ask("\n[bold green]You[/bold green]")
        if user_input.lower() in ['exit', 'quit']:
            console.print("[bold blue]Goodbye![/bold blue]")
            break
        if not user_input:
            continue
            
        state["messages"].append(HumanMessage(content=user_input))
        
        try:
            console.print("\n[bold purple]Agent:[/bold purple]")
            
            full_response = ""
            pre_run_message_count = len(state["messages"])
            
            # Use stream_mode=["messages", "values"] to get both token streaming and state
            # in a SINGLE graph execution run, preventing duplicate tool executions!
            final_state = state
            with Live(console=console, refresh_per_second=30) as live:
                for stream_mode, chunk in app.stream(state, stream_mode=["messages", "values"]):
                    if stream_mode == "messages":
                        msg_chunk, metadata = chunk
                        if isinstance(msg_chunk, AIMessageChunk):
                            # Intercept tool calls for real-time observability
                            if hasattr(msg_chunk, "tool_call_chunks") and msg_chunk.tool_call_chunks:
                                for tool_call in msg_chunk.tool_call_chunks:
                                    if tool_call.get("name"):
                                        live.console.print(f"[bold dim cyan]🛠️  Agent is using {tool_call['name']}...[/bold dim cyan]")

                            if msg_chunk.content:
                                # Stream chunks from the LLM node (legacy name 'agent' kept for compatibility)
                                if metadata.get("langgraph_node") in {"call_llm", "agent"}:
                                    full_response += msg_chunk.content
                                    live.update(Markdown(full_response))
                    elif stream_mode == "values":
                        # chunk contains the entire state dict
                        final_state = chunk
                        
            state = final_state

            # Fallback: if no chunks were rendered, print the final AI message from state.
            if not full_response.strip():
                fallback_response = ""
                for msg in reversed(state.get("messages", [])[pre_run_message_count:]):
                    if isinstance(msg, AIMessage) and msg.content:
                        fallback_response = msg.content if isinstance(msg.content, str) else str(msg.content)
                        break

                if fallback_response:
                    console.print(Markdown(fallback_response))
                else:
                    console.print("[dim]No response generated.[/dim]")
            
        except Exception as e:
            console.print(f"\n[bold red][System Error]:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()


