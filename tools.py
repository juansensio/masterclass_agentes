from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# tools

tool = TavilySearch(max_results=3)

# llm

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3, 
    max_tokens=1000,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

llm_with_tools = llm.bind_tools([tool])

# grafo

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools=[tool])

graph_builder = StateGraph(State)
graph_builder.add_edge(START, "agent")
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()

# ejecución

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()