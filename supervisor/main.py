from langchain_groq import ChatGroq
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import json

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
        print(f"Update from subgraph {graph_id}:\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print()

        # ✅ Add this null check
        if node_update is None or "messages" not in node_update:
            print("\t(No messages returned or node failed)\n")
            continue

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print()


load_dotenv()

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0) #,max_tokens=4000)

# query_analyzer -> extract trip details from user input
# hotel_agent
# weather_agent
# attractions_agent
# calculator_agent
# iternary_agent -> build day by day plan
# summary_agent

##########################################################################################
def weather_tool(query : str):
    """provide weather details of the places to travel"""
    response = model.invoke(query)
    return response.content[:1000]


weather_agent = create_react_agent(
    model=model,
    tools=[weather_tool],
    name="weather_expert",
    prompt = """You are an weather agent . You have web access . Provide weather related information.
                Do not do anything else other than reporting about weather conditions.
                
                As a weather_agent, I need to gather specific information to provide accurate and helpful 
                weather details, precautions, and packing recommendations for travelers.
        
                First using the provided query extract the place and date for which weather information is required.
                
                Please provide the following details:

                1.  **Destination:** What is the name of the city or place you are traveling to? (e.g., Paris, Kyoto, New York City, Goa)

                2.  **Travel Dates:**
                    * Are you traveling on a specific date (e.g., "August 15th, 2025")?
                    * Are you traveling for a period (e.g., "from July 1st to July 7th, 2025")?
                    * Or is it a general time frame (e.g., "next week", "this coming weekend", "in three days")?
                    * *If no dates are specified, I will assume you are asking for the weather over the next 7 days from today's date.*

                Once I have this information, I will provide:
                * The general temperature range and typical conditions for that time of year.
                * A detailed daily forecast for your specific travel dates.
                * Recommended precautions for the expected weather (e.g., staying hydrated in heat, layering in cold, rain gear).
                * A suggested packing list tailored to the weather conditions.

                Example Input:
                "I'm planning a trip to London from December 20th to December 27th, 2025."
                "What's the weather like in Tokyo next month?"
                "Tell me about the weather in Delhi on July 10th."  

                Here is the user's query:
                {query}
                """
)

##########################################################################################


def hotel_searching_tool(query: str):
    """searches hotels for travellig"""
    response = model.invoke(query)

    # Take the content and ensure it is returned as a plain string
    content = response.content

    try:
        # Try parsing as JSON to format cleanly (optional)
        hotel_data = json.loads(content)
        pretty_str = json.dumps(hotel_data, indent=2)
        return f"HOTEL JSON (Plain Text):\n{pretty_str[:1000]}"
    except Exception:
        # If not JSON, return raw
        return f"HOTEL RAW:\n{content[:1000]}"


hotel_agent = create_react_agent(
    model=model,
    tools=[hotel_searching_tool],
    name="hotel_expert",
    prompt="""
You are a hotel_searching_agent that helps users find hotel options.

✳️ STRICTLY FOLLOW THIS FORMAT:
Return your result as a **JSON object** with a list of at most 2 hotels.

Example output format:
{
  "hotels": [
    {
      "name": "Hotel Sunshine",
      "address": "123 Main St, City",
      "star_rating": "3-star",
      "price_per_night": "₹2500",
      "total_price": "₹10000",
      "amenities": ["Wi-Fi", "Breakfast"],
      "type": "Budget",
      "note": "Close to city center"
    },
    {
      "name": "Luxury Stay Inn",
      "address": "456 View Point, Hill Area",
      "star_rating": "4-star",
      "price_per_night": "₹6000",
      "total_price": "₹24000",
      "amenities": ["Wi-Fi", "Spa", "Mountain view"],
      "type": "Luxury",
      "note": "Perfect for couples"
    }
  ]
}

✳️ Do NOT use markdown.
✳️ Do NOT include hotel links or HTML.
✳️ Do NOT return more than 2 hotels.
✳️ Keep total output under 1000 characters.

---

Start by extracting these from user input:
1. Destination
2. Travel Dates
3. Guests
4. Budget
5. Room & Amenity Preferences
6. Hotel Type (Budget, Mid-range, Luxury)

Then return hotel results **only in the format above**.

"""
)

##########################################################################################

def attractions_tool(query: str):
    """
    This function uses an attractions_agent to extract the destination city or place from the user's query.
    It then returns a comprehensive guide to the top attractions in that location, including pricing, transport options, and practical tips.
    """
    response = model.invoke(query)
    return response.content[:1000]


attractions_agent = create_react_agent(
    model=model,
    tools=[attractions_tool],
    name="attractions_expert",
    prompt="""
           You are an attractions agents. You have web access.
           Provide details about local tourist attractions and tourist spots to visit.

           You are an attractions_agent that helps travelers explore the best local tourist attractions based on their travel destination.

           Start by extracting the **destination** from the user's query and then provide a helpful travel guide tailored to that location.

            ---

            ### Provide the following details in your response:

            1. **Top 5–10 Tourist Attractions in {user_query or "the mentioned destination"}**:
            - Name of the attraction
            - Brief description (history, uniqueness, type – natural/cultural/religious/fun)
            - Entry Fee (mention "Free" if no charges, otherwise give local currency)
            - Mention distance from city.
            - Best time of day or season to visit
            - Approximate duration of visit (e.g., 1 hr, half-day, full-day)
            - Opening and closing hours (if applicable)
            - Local popularity or rating (optional)

            2. **How to Reach Each Attraction**:
            - Common modes of transport (e.g., local bus, metro, tuk-tuk, taxi, bike rental, walking)
            - Estimated travel time from the city center or airport
            - Approximate cost for each mode (in local currency)
            - Any tips or known issues (e.g., traffic congestion, no direct public transport, needs hike)

            3. **Do’s and Don’ts** for Tourists:
            - Respectful behavior in religious or cultural places (dress code, silence, removing shoes, etc.)
            - Photography restrictions
            - Safety precautions (e.g., wildlife, scams, altitude, strong sun)
            - Environmental responsibility (e.g., no littering, plastic bans)
            - Local customs or etiquette tourists should follow

            4. **Local Tips or Packages** (if available):
            - Combo passes or city cards that offer discounts for multiple attractions
            - Free walking tours or local guide recommendations
            - Offbeat or lesser-known places to explore nearby
            - Local festivals or events (if timing matches)

            ---

            ### Example Inputs:
            - "What are the best places to visit in Udaipur?"
            - "I'm planning to travel to Kyoto next month. What attractions should I not miss?"
            - "Things to see in Rome including entry fees and travel costs."
            - "Tell me about tourist places in Singapore and how to get there."      
           """
)

##########################################################################################

workflow = create_supervisor(
    model=model,
    agents=[weather_agent, hotel_agent, attractions_agent],
    prompt="""
You are a supervisor agent managing 3 agents:
- **weather_agent**: for weather forecast, packing suggestions, and precautions
- **hotel_agent**: for finding hotel options, pricing, and amenities
- **attractions_agent**: for identifying top local attractions, prices, and travel tips

---

Your task is to:
1. **Sequentially call** the 3 agents based on the user query.
2. **Collect and store** their outputs internally.
3. Once all three agents have responded:
   - **Synthesize a complete day-wise travel itinerary** that includes:
     - Recommended attractions per day (based on weather)
     - Suggested hotel options and check-in/out times
     - Packing advice and precautions
     - Local travel tips and costs

---

✳️ Do NOT assign work to multiple agents at once.  
✳️ Do NOT answer directly before gathering all necessary info.  
✳️ Do NOT ask the user again for info already provided.

Return a **well-formatted, final itinerary** only after gathering:
- Weather data
- Hotel data
- Attractions data
""",
    add_handoff_back_messages=True,
    output_mode="full_history"
)


app = workflow.compile()


image = app.get_graph().draw_mermaid_png()
with open('supervisor-agent.png','wb') as f:
    f.write(image)


user_input  = ""

# Define the config for the stream method
stream_config = {"recursion_limit": 10}

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