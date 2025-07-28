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
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# memoria

config = {"configurable": {"thread_id": "1"}}

checkpointer = MemorySaver()
# checkpointer = SqliteSaver(conn=sqlite3.connect("agente.db", check_same_thread=False))

# tools

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

@tool
def linkedin_search(search_terms: str) -> str:
    """
    Search for LinkedIn profiles using specific industry terms, job titles, or company types.
    
    Args:
        search_terms: Specific keywords, job titles, industries, or company types to search for
                     (e.g., "restaurant industry CEO", "pet services entrepreneur", "fintech startup founder")
    """
    tavily = TavilySearch(max_results=5)
    
    # Search for both individual profiles and companies
    linkedin_query = f"site:linkedin.com/in {search_terms} OR site:linkedin.com/company {search_terms}"
    
    try:
        # Get results from Tavily
        search_results = tavily.run(linkedin_query)
        
        # Handle different response formats from Tavily
        if isinstance(search_results, str):
            # If Tavily returns a string, parse it or return as is
            return f"LinkedIn search results for '{search_terms}':\n\n{search_results}"
        
        if isinstance(search_results, list):
            results = search_results
        elif isinstance(search_results, dict) and 'results' in search_results:
            results = search_results['results']
        else:
            return f"Unexpected response format from search for: {search_terms}"
        
        if not results:
            return f"No LinkedIn profiles found for: {search_terms}"
        
        formatted_results = f"LinkedIn profiles found for '{search_terms}':\n\n"
        
        # Process up to 5 results
        for i, result in enumerate(results[:5], 1):
            if isinstance(result, dict):
                url = result.get('url', 'No URL')
                title = result.get('title', 'No title')
                content = result.get('content', result.get('snippet', 'No description available'))
            else:
                # If result is not a dict, convert to string
                url = 'No URL'
                title = f'Result {i}'
                content = str(result)
            
            # Clean up the content to make it more readable
            content_preview = content[:200] + "..." if len(content) > 200 else content
            
            formatted_results += f"{i}. **{title}**\n"
            formatted_results += f"   URL: {url}\n"
            formatted_results += f"   Description: {content_preview}\n\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Error searching LinkedIn profiles: {str(e)}"

tools = [
    human_assistance,
    linkedin_search,
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

def agent1(state: State):
    system_prompt = """
        Eres un agente especializado en validar ideas de negocio. Tu tarea es:
        1. Recibir una business idea del usuario
        2. Analizar la idea para identificar industrias, roles y tipos de empresas relevantes para la validación de la idea (deben ser potenciales clientes)

        Cuando recibas una business idea:

        PASO 1: Analiza la idea de negocio e identifica:
        - ¿Qué industrias son potenciales clientes?
        - ¿Qué roles/posiciones son potenciales clientes?
        - ¿Qué tipos de empresas son potenciales clientes?
        - ¿Qué expertise específico sería valioso para la validación de la idea?

        PASO 2: Forma consultas de búsqueda específicas y targeted para LinkedIn:
        - En lugar de buscar la idea completa, busca términos específicos como:
          * "restaurant industry CEO" 
          * "pet services entrepreneur"
          * "fintech startup founder"
          * "healthcare technology director"
        
        No uses ninguna herramienta, sólo responde con la información que has encontrado.
    """
    conversation = [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ]
    return {"messages": [llm_with_tools.invoke(conversation)]}

def agent2(state: State):
    system_prompt = """
        Eres un agente especializado en validar ideas de negocio. Tu tarea es:
        1. Recibir consultas de búsqueda específicas de linkedin para validar una business idea
        2. Buscar perfiles de LinkedIn estratégicos que puedan ayudar a validar la idea
        3. Devolver el resultado de la búsqueda

        Cuando recibas una consulta de búsqueda específica de linkedin:

        PASO 1: Usa linkedin_search con los términos específicos que identificaste 

        PASO 2: Para cada perfil encontrado, crea un mensaje de introducción personalizado que se adapte a la industria, posición y empresa del perfil así como una lista de 3 preguntas que se pueden hacer para validar la idea.

        Ejemplo:
        Business idea: "Una app que ayuda a encontrar restaurantes pet-friendly"
        Búsqueda: "restaurant industry executives" (no "Una app que ayuda a encontrar restaurantes pet-friendly")
        Resultado:
        - Perfil 1: "John Doe, CEO of Pet Friendly Restaurants"
            - Mensaje de introducción: "Hola John, me llamo [Tu nombre] y soy [Tu rol]. Estoy trabajando en una startup que busca validar la idea de una app que ayuda a encontrar restaurantes pet-friendly. ¿Te gustaría saber más sobre la app y cómo podría ayudarte?"
            - Lista de 3 preguntas:
            * ¿Qué te parece la idea de una app que ayuda a encontrar restaurantes pet-friendly?
            * ¿Qué te parece la idea de una app que ayuda a encontrar restaurantes pet-friendly?
            * ¿Qué te parece la idea de una app que ayuda a encontrar restaurantes pet-friendly?
        - Perfil 2: "Jane Smith, Founder of Pet Services"
            - Mensaje de introducción: "Hola Jane, me llamo [Tu nombre] y soy [Tu rol]. Estoy trabajando en una startup que busca validar la idea de una app que ayuda a encontrar restaurantes pet-friendly. ¿Te gustaría saber más sobre la app y cómo podría ayudarte?"
            - Lista de 3 preguntas:
            * ¿Qué te parece la idea de una app que ayuda a encontrar restaurantes pet-friendly?
            * ¿Qué te parece la idea de una app que ayuda a encontrar restaurantes pet-friendly?
            * ¿Qué te parece la idea de una app que ayuda a encontrar restaurantes pet-friendly?
        Sé proactivo y ejecuta las herramientas necesarias sin esperar confirmación.
    """
    conversation = [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ]
    return {"messages": [llm_with_tools.invoke(conversation)]}

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)
graph_builder.add_edge(START, "agent1")
graph_builder.add_node("agent1", agent1)
graph_builder.add_node("agent2", agent2)
graph_builder.add_edge("agent1", "agent2")
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("agent2", tools_condition)
graph_builder.add_edge("tools", "agent2")
graph = graph_builder.compile(checkpointer=checkpointer)
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

# ejecución

while True:
    business_idea = input("💡 Business Idea: ")
    if business_idea.lower() in ["quit", "exit", "q"]:
        print("Goodbye! 👋")
        break
    events = graph.stream(
        {"messages": [{"role": "user", "content": business_idea}]},
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