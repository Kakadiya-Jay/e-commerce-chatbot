import streamlit as st
from router import router
from faq import faq_chain, ingest_faq_data
from pathlib import Path
from sql import sql_chain  

faqs_path = Path(__file__).parent / "resources/faq_data.csv"
ingest_faq_data(faqs_path)


def ask(query: str):
    selected_route = router(query).name
    if selected_route == "FAQ Route":
        return faq_chain(query)
    elif selected_route == "SQL Route":
        return sql_chain(query)
    else:
        return "Sorry, I can only handle FAQ queries at the moment."


st.title("Welcome to My Streamlit App")

query = st.chat_input("Write your message here...")

st.session_state["messages"] = st.session_state.get("messages", [])

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query:
    st.chat_message("user").markdown(query)
    st.session_state["messages"].append({"role": "user", "content": query})
    # Here you can add logic to process the query and generate a response
    response = ask(query)
    st.chat_message("assistant").markdown(response)
    st.session_state["messages"].append({"role": "assistant", "content": response})
