from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain.tools import tool
# from langgraph_supervisor import create_supervisor


load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, max_tokens=500) #,max_tokens=4000)

## history question answer tool

def history(question: str):
    """answer question related to history"""
    print("[HISTORY TOOL] Received question:", question)
    response = llm.invoke(question)
    return response.content

## science and technology question answer tool

def science(question: str):
    """answer question related to science and technology"""
    print("[SCIENCE TOOL] Received question:", question)
    response = llm.invoke(question)
    return response.content

## stock market question answer tool

def market(question: str):
    """answer question related to finance and stock market"""
    print("[MARKET TOOL] Received question:", question)
    response = llm.invoke(question)
    return response.content

## history question answering agent

history_agent = create_react_agent(
    model=llm,
    tools = [history,create_handoff_tool(agent_name="science_agent"),create_handoff_tool(agent_name="market_agent")],
    name = "history_agent",
    prompt = (
        """
        You are a historian who ONLY answers history related question.
        If a question is not clearly related to history, do NOT answer it.
        For any question related to science and technology topics,
        you must use handoff tool to transfer the question to science agent.
        For any question related to stock market related topics, you must use
        handoff tool to transfer question to market agent.
        Use the handoff tool ONLY ONCE if confident it's about science or finance.
        Avoid circular handoffs. Respond with "This question doesn't seem to relate to history" if unsure.
        """
    )
)

market_agent = create_react_agent(
    model=llm,
    tools = [market,create_handoff_tool(agent_name="science_agent"),create_handoff_tool(agent_name="history_agent")],
    name = "market_agent",
    prompt = (
        """
        You are a financial expert.
        Answer ONLY finance and stock market-related questions.
        For any question related to science and technology topics,
        you must use handoff tool to transfer the question to science agent.
        For any question related to history related topics, you must use
        handoff tool to transfer question to history agent.
        Use the handoff tool ONLY ONCE if confident it's about history or science.
        Avoid recursive handoffs. If unsure, state the question is outside your domain.
        """
    )
)

science_agent = create_react_agent(
    model=llm,
    tools = [science,create_handoff_tool(agent_name="history_agent"),create_handoff_tool(agent_name="market_agent")],
    name = "science_agent",
    prompt = (
        """
        You are a science and technology expert.
        Only answer questions clearly related to science or technology.
        For any question related to history topics,
        you must use handoff tool to transfer the question to history agent.
        For any question related to stock market related topics, you must use
        handoff tool to transfer question to market agent.
        Use the handoff tool ONLY ONCE if confident it's about history or finance.
        Avoid infinite handoff loops. If uncertain, respond that it's outside your scope.
        """
    )
)

# initialize the supervisor and application
# checkpoint = InMemorySaver()
swarm = create_swarm(
    agents = [history_agent,market_agent,science_agent],
    default_active_agent="science_agent",
)

app = swarm.compile()   #checkpointer=checkpoint

# Save the workflow diagram
image = app.get_graph().draw_mermaid_png()
with open("swarm-agent.png", "wb") as f:
    f.write(image)

# Configuration for the conversation thread
config = {"configurable": {"thread_id": "1"}}


def display_result(result):
    print("\n=== RESPONSE ===")
    if "agent_name" in result:
        print(f"Agent: {result['agent_name']}")
    for msg in result["messages"]:
        role = msg.__class__.__name__.replace("Message", "").upper()
        print(f"{role}: {msg.content}")
    print("================\n")

config = {
    "configurable": {
        "thread_id": "1"
    },
    "recursion_limit": 50
}

while True:
    user_input = input("\nEnter your request (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:
        result = app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
        )
        display_result(result)  # ✅ FORMATTED OUTPUT
    except Exception as e:
        print("[ERROR] Something went wrong:", e)  # ✅ ERROR HANDLING