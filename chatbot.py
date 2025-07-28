from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

# llm

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3, 
    max_tokens=1000,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# grafo

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("chatbot", END)
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