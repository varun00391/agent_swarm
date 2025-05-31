from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

load_dotenv()

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0) #,max_tokens=4000)

def add(a,b):
    """add two numbers"""
    return float(a)+float(b)

def multiply(a,b):
    """multiply two numbers"""
    return float(a)*float(b)

def web_search(query: str) -> str:
    """search web for information"""
    web_search = TavilySearch(max_results=3)
    results = web_search.invoke(query)

    if results and results[0] and "content" in results[0]:
        return results[0]["content"]
    return "No relevant information found"


math_agent = create_react_agent(
    model = model,
    tools = [add,multiply],
    name = "math_expert",
    prompt = """
             You are a math export. Always use one tool at a time to solve problem.
             If you are given text, you must first extract the numbers needed for calculations.
             """
)


research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="""
           You are a world class researcher with access to web search.
           Your goal is to find relevant information based on query.
           When the query involves numerical calculations (like sum, average, etc.),
           you *must* extract the relevant numbers from the search results and present them clearly.
           For example, if asked for GDPs, find them and list them out.
           Do not perform any math yourself.
           """
)


workflow = create_supervisor(
    model=model,
    agents=[research_agent, math_agent],
    prompt=(
        """
        You are a supervisor managing two agents: a research agent, a math agent.
        - **research_agent**: Use for finding raw information from the web and extracting relevant numerical data.
        - **math_agent**: Use for performing calculations on numbers.
        Assign work to one agent at a time, do not call agents in parallel. Do not do any work yourself.
        Ensure data is extracted before math is attempted.

        Your goal is to answer the user's query. Once the final answer is computed,
        return the answer directly to the user. Do not re-assign if the final answer is available.
        """
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
)


app = workflow.compile()


image = app.get_graph().draw_mermaid_png()
with open('supervisor-agent.png','wb') as f:
    f.write(image)

user_input  = "Find the difference between the highest mountain in Africa and the highest mountain in North America (in meters)."

# Define the config for the stream method
stream_config = {"recursion_limit": 50}

for chunk in app.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": user_input,
            }
        ]
    },
    config = stream_config
):
    pretty_print_messages(chunk, last_message=True)

final_message_history = chunk["supervisor"]["messages"]