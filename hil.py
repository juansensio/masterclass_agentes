from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
import sys

load_dotenv()

# memoria

config = {"configurable": {"thread_id": "3"}}

# checkpointer = MemorySaver()
checkpointer = SqliteSaver(conn=sqlite3.connect("chat.db", check_same_thread=False))

# tools

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tools = [
    TavilySearch(max_results=3),
    human_assistance,
]

# llm

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3, 
    max_tokens=1000,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

llm_with_tools = llm.bind_tools(tools)

# grafo

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)
graph_builder.add_edge(START, "agent")
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile(checkpointer=checkpointer)

# ejecución

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config, # para seguir un hilo de conversación
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    # Check if we need human input for a tool
    while True:
        snapshot = graph.get_state(config)
        if snapshot.next and snapshot.next[0] == 'tools':
            # Tool is waiting for human input
            human_response = input("Human: ")
            if human_response.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                sys.exit(0)
            # Resume the tool execution with human response
            human_command = Command(resume={"data": human_response})
            events = graph.stream(human_command, config, stream_mode="values")
            # Process the response after human input
            for event in events:
                if "messages" in event:
                    event["messages"][-1].pretty_print()
        else:
            # No more tools waiting, break out of the inner loop
            break