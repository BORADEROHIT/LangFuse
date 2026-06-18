from __future__ import annotations
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langfuse.langchain import CallbackHandler
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
#from datetime import datetime
import random
from langchain_ollama import ChatOllama
from typing import TypedDict, Literal, Optional, Dict, Any


# =========================================================
# 0) Langfuse + LLM
# =========================================================
langfuse_handler = CallbackHandler()
llm = ChatOllama(model="llama3.1",temperature=0.2)

# =========================================================
# 1) State for LangGraph
# =========================================================
class AgentState(TypedDict):
    query: str              # user query
    route: Optional[Literal["research", "utility"]]  # where supervisor sends it
    result: Optional[str]   # final answer from the chosen agent
    meta: Dict[str, Any]    # debugging / traces / intermediate info


# =========================================================
# 2) Tools
# =========================================================

# Wikipedia tool
wiki = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="wikipedia",
    func=wiki.run,
    description="Search Wikipedia and return relevant information given a user query.",
)

# --- Default 'utility' tool:  ---
def get_random_number(_):
    return f"Your random number is: {random.randint(1, 10)}"

random_tool = Tool(
    name="random_number",
    func=get_random_number,
    description="Returns a random number between 1 and 100"
)

utility_tool = random_tool


# =========================================================
# 3) Supervisor (LLM-based router)
#    Decides whether to use "research" or "utility"
# =========================================================
SUPERVISOR_SYSTEM = """You are a router. Decide if the user query should be answered by:
- 'research' (uses Wikipedia to look up facts/topics),
- 'utility' (uses a system utility tool, e.g., random number).

Return ONLY one token: 'research' or 'utility'.
"""

def supervisor_node(state: AgentState) -> AgentState:
    query = state["query"]

    # Ask the LLM to choose a route (research vs utility)
    routing_prompt = [
        {"role": "system", "content": SUPERVISOR_SYSTEM},
        {"role": "user", "content": f"User query: {query}\nWhich route? (research|utility)"},
    ]
    resp = llm.invoke(routing_prompt)
    route_raw = (resp.content or "").strip().lower()

    # Simple normalization/guard
    route: Literal["research", "utility"] = "research"
    if "utility" in route_raw:
        route = "utility"
    else:
        route = "research"

    state["route"] = route
    state.setdefault("meta", {})
    state["meta"]["route_raw"] = route_raw
    return state


# =========================================================
# 4) Research Agent Node
#    Calls the Wikipedia tool directly (no initialize_agent)
# =========================================================
def research_node(state: AgentState) -> AgentState:
    query = state["query"]
    # Run the wikipedia tool directly
    wiki_result = wikipedia_tool.run(query)
    # Optionally let the LLM polish/summarize the tool output
    prompt = [
        {"role": "system", "content": "You are a helpful research assistant. Concise and accurate."},
        {"role": "user", "content": f"Question: {query}\n\nWikipedia says:\n{wiki_result}\n\nWrite a concise answer."},
    ]
    resp = llm.invoke(prompt)
    state["result"] = resp.content
    return state


# =========================================================
# 5) Utility Agent Node
#    Calls the utility tool (default: current time)
# =========================================================
def utility_node(state: AgentState) -> AgentState:
    query = state["query"]

    # We can allow the LLM to decide how to call the tool, or just call directly.
    # Here, call directly (the tool ignores input for current time).
    util_value = utility_tool.run(query)

    # Make the answer user-friendly via the LLM (optional)
    prompt = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": f"User asked: {query}\nUtility tool returned: {util_value}\nRespond helpfully."},
    ]
    resp = llm.invoke(prompt)
    state["result"] = resp.content
    return state


# =========================================================
# 6) Build LangGraph
# =========================================================
graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("research_agent", research_node)
graph.add_node("utility_agent", utility_node)

# Entry point
graph.set_entry_point("supervisor")

# Conditional edges based on supervisor decision
def route_decider(state: AgentState) -> str:
    return "research_agent" if state.get("route") == "research" else "utility_agent"

graph.add_conditional_edges("supervisor", route_decider)
graph.add_edge("research_agent", END)
graph.add_edge("utility_agent", END)

app = graph.compile()


# =========================================================
# 7) Run examples
# =========================================================
if __name__ == "__main__":
    print("\n--- Test  (Utility) ---")
    out2 = app.invoke(
        {"query": "Give me a random number"},
        config={"callbacks": [langfuse_handler]},
    )
    print(out2["result"])

    print("\n--- Test 2 (Research) ---")
    out1 = app.invoke(
        {"query": "Summarize the Turing test in a sentence."},
        # You can attach callbacks at call-time as well; they merge with LLM callbacks.
        config={"callbacks": [langfuse_handler]},
    )
    print(out1["result"])