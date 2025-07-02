# import streamlit as st
# from main import app, pretty_print_messages  # Import LangGraph app & printer
# from langchain_core.messages import convert_to_messages

# st.set_page_config(page_title="Smart Travel Itinerary Planner", layout="centered")

# st.title("🗺️ Travel Itinerary Planner")
# st.markdown("Enter your travel query below to get a detailed itinerary:")

# user_input = st.text_area("✈️ Where do you want to go?", height=100, placeholder="E.g. Plan a trip to Gangtok from 9th July to 13 July")

# submit = st.button("Generate Itinerary")

# if submit and user_input:
#     with st.spinner("🧠 Planning your trip..."):
#         final_response = ""
#         stream_config = {"recursion_limit": 10}

#         final_response = ""

#         # Accumulate the final supervisor message safely
#         for chunk in app.stream(
#             {
#                 "messages": [{"role": "user", "content": user_input}]
#             },
#             config={"recursion_limit": 10}
#         ):
#             if "supervisor" in chunk and "messages" in chunk["supervisor"]:
#                 final_messages = chunk["supervisor"]["messages"]
#                 final_response = final_messages[-1].content

#         # for chunk in app.stream(
#         #     {
#         #         "messages": [{"role": "user", "content": user_input}]
#         #     },
#         #     config=stream_config
#         # ):
#         #     # Get the latest message (agent output) for rendering
#         #     final_msg = chunk["supervisor"]["messages"][-1]
#         #     final_response = final_msg.content  # Update final content

#             # Optional: Live streaming output (can be removed)
#             with st.expander("🧵 Debug Messages", expanded=False):
#                 pretty_print_messages(chunk, last_message=True)

#         # Final Result
#         st.markdown("---")
#         st.subheader("🧳 Your Trip Itinerary")
#         st.markdown(final_response)


import streamlit as st
from PIL import Image
import streamlit as st
from main import app, pretty_print_messages  # Import LangGraph app & printer
# from langchain_core.messages import convert_to_messages


st.set_page_config(page_title="Smart Travel Itinerary Planner", layout="centered")

# Display the agent image in the sidebar
with st.sidebar:
    st.markdown("### 🧠 Agent Workflow")
    image = Image.open("supervisor-agent.png")
    st.image(image, caption="Agent Workflow", width=400)

# Main Title and Input
st.title("🗺️ Travel Itinerary Planner")
st.markdown("Enter your travel query below to get a detailed itinerary:")

user_input = st.text_area(
    "✈️ Where do you want to go?",
    height=100,
    placeholder="E.g. Plan a trip to Gangtok from 9th July to 13 July"
)

submit = st.button("Generate Itinerary")

if submit and user_input:
    with st.spinner("🧠 Planning your trip..."):
        final_response = ""
        stream_config = {"recursion_limit": 10}

        for chunk in app.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=stream_config
        ):
            if "supervisor" in chunk and "messages" in chunk["supervisor"]:
                final_messages = chunk["supervisor"]["messages"]
                final_response = final_messages[-1].content

            with st.expander("🧵 Debug Messages", expanded=False):
                pretty_print_messages(chunk, last_message=True)

    # Display the final itinerary
    st.markdown("---")
    st.subheader("🧳 Your Trip Itinerary")
    st.markdown(final_response)
