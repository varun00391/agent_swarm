# Smart Travel Itinerary Planner (LangGraph + Groq)

This project is a **multi-agent travel planning assistant** powered by **LangGraph**, **LangChain**, and **Groq LLMs**. It intelligently builds a day-by-day itinerary using separate agents for weather, hotels, and local attractions.

---

## Key Components

### 1. Groq LLM
- Uses `meta-llama/llama-4-scout-17b` for fast and accurate results.

### 2. Agents

- **weather_agent**  
  Provides weather forecasts, precautions, and packing tips.

- **hotel_agent**  
  Finds 1–2 hotel options based on destination, dates, budget, and preferences.  
  Returns results in a strict JSON format.

- **attractions_agent**  
  Suggests top tourist spots, local travel info, entry costs, and cultural tips.

---

### 3. Supervisor Agent

- **Manages the three agents sequentially**
- Collects responses from each one
- Then composes a **final day-wise itinerary** that includes:
  - Daily attraction suggestions (weather-aware)
  - Hotel recommendations
  - Packing list & weather precautions
  - Local travel tips and transport costs

---

### 4. Visualization and Debugging

- Generates a **Mermaid diagram** of the agent workflow and saves it as `supervisor-agent.png`
- Streams detailed agent messages for live debug/inspection
- Can be used in a **Streamlit UI** with a sidebar image display

---

##  What This Code Does

- Loads environment variables using `.env`
- Initializes Groq model using `ChatGroq`
- Defines tool functions for:
  - Weather data
  - Hotel search
  - Attractions info
- Creates `create_react_agent` agents with structured prompts
- Builds a `create_supervisor` workflow to orchestrate them
- Streams final itinerary step-by-step and saves visual graph

---

##  Sample Output Includes:

- **Weather forecast** with daily breakdown and packing suggestions
- **Up to 2 curated hotels** with amenities, pricing, and notes
- **Top 5–10 attractions** with travel tips, fees, and how to reach
- **Complete day-wise travel itinerary** for the trip

---

## 🖼️Mermaid Graph

To visualize the agent interactions, the graph is generated using:

```python
image = app.get_graph().draw_mermaid_png()
with open('supervisor-agent.png', 'wb') as f:
    f.write(image)
```
-----------
