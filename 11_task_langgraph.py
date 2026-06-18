from dotenv import load_dotenv
load_dotenv()
# -----------------------------------------
# 1. Import required libraries
# -----------------------------------------
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama  # Ollama LLM integration
from langchain.tools import tool         # For defining tools
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Langfuse for observability
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# -----------------------------------------
# 2. Initialize Langfuse client and handler
# -----------------------------------------
langfuse = get_client()  # Reads keys from environment variables
langfuse_handler = CallbackHandler()  # Enables tracing for LangChain/LangGraph

# -----------------------------------------
# 3. Define a single tool: Calculator
# -----------------------------------------
@tool
def calculator(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression."""
    try:
        # Allow only digits and basic operators
        safe = "".join(ch for ch in expression if ch in "0123456789+-*/(). ")
        result = eval(safe)  # Evaluate the expression
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Create a ToolNode for LangGraph
tools = [calculator]
tool_node = ToolNode(tools)

# -----------------------------------------
# 4. Define the graph state
# -----------------------------------------
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]  # Stores conversation messages

# -----------------------------------------
# 5. Initialize Ollama LLM and bind tools
# -----------------------------------------
# Make sure Ollama is running and model (e.g., llama3.1) is pulled
llm = ChatOllama(model="llama3.1", temperature=0).bind_tools(tools)

# -----------------------------------------
# 6. Define agent logic
# -----------------------------------------
def should_continue(state: MessagesState) -> Literal["tools", END]:
    """Check if the AI requested a tool; if yes, go to tools, else end."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

def agent_node(state: MessagesState) -> dict:
    """Agent reasoning step: generate next AI message using Ollama."""
    messages = state["messages"]
    sys_msg = SystemMessage(content="You are a helpful assistant. Use the calculator tool if needed.")
    response = llm.invoke([sys_msg] + messages)  # LLM call traced by Langfuse
    return {"messages": [response]}

# -----------------------------------------
# 7. Build the LangGraph workflow
# -----------------------------------------
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge("__start__", "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()

# -----------------------------------------
# 8. Run the agent with Langfuse tracing
# -----------------------------------------
if __name__ == "__main__":
    user_input = "What is (42 + 58) * 2? and zhu yi writer belongs to which country?"
    
    # Invoke the graph with Langfuse callback for observability
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"callbacks": [langfuse_handler]},
    )

    # Flush events to Langfuse (important for short scripts)
    langfuse.flush()

    # Print final AI response
    final_msg = [m for m in result["messages"] if isinstance(m, AIMessage)]
    print("\nFinal Answer:\n", final_msg[-1].content if final_msg else "(No AIMessage)")

