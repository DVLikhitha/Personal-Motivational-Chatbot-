import streamlit as st
from datetime import datetime
import random
from nlp_model import model, tokenizer, pad_sequences, X, lbl_enc, df
import re

# Function to set Streamlit page configuration
def setup_page():
    st.set_page_config(page_title="Chatbot for Depression Support", page_icon="💬")

# Function to generate response
def generate_answer(user_input):
    pattern = user_input
    if pattern.lower() == 'quit':
        return "Goodbye!"

    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)

    x_test = tokenizer.texts_to_sequences(text)
    x_test = pad_sequences(x_test, padding='post', maxlen=X.shape[1])

    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()

    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    return random.choice(responses)

# Function to generate and add messages to chat history
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# Function to save current chat session and clear messages
def save_and_clear_chat():
    session_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_sessions.append({"name": session_name, "messages": st.session_state.messages.copy()})
    st.session_state.messages = []

# Function to refresh chat to a new page
def refresh_chat():
    session_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_sessions.append({"name": session_name, "messages": st.session_state.messages.copy()})
    st.session_state.messages = []
    st.rerun()

# Function to display previous chat sessions
def display_previous_chats():
    st.sidebar.header("Previous Chats")
    for i, session in enumerate(st.session_state.chat_sessions):
        if st.sidebar.button(f"Chat from {session['name']}", key=f"chat_{i}"):
            st.session_state.messages = session["messages"]
            st.rerun()

# Initialize Streamlit page configuration
setup_page()

# Initialize chat history and sessions in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

# Display chat messages
st.title("Chatbot for Depression Support")
st.subheader("Chat with the bot")

# Actions buttons at the top-left
  # Add a separator
if st.sidebar.button("Refresh Chat", key="refresh_chat"):
    refresh_chat()

# Display previous chat sessions (in the left sidebar)
display_previous_chats()

# Scrollable chat container
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<p style="color: yellow;">👤You:</p> {message["content"]}', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color: orange;">🤖 Bot:</p> {message["content"]}', unsafe_allow_html=True)

# User input handling
def submit_message():
    user_input = st.session_state.user_input
    add_message("user", user_input)
    response = generate_answer(user_input)
    add_message("bot", response)
    st.session_state.user_input = ""
    st.rerun()

# Fixed position user input at the bottom
st.markdown("---")
input_container = st.empty()
with input_container:
    user_input = st.text_input("Type your message here...", key="user_input", on_change=submit_message)

# Actions buttons
st.markdown("---")

# Clear chat history button
if st.button("Clear Chat History"):
    save_and_clear_chat()

# Predefined questions asked by depressed people
st.markdown("### Common Questions Asked by Depressed People")

# Questions as buttons in columns
col1, col2, col3 = st.columns(3)
questions = [
    "How can I get help for my depression?",
    "How can I cope with my anxiety?",
    "What are some tips for managing stress?",
    "How can I improve my sleep?",
    "Why do I feel so tired all the time?",
    "What are the side effects of antidepressants?",
    "How do I know if I need to see a therapist?",
    "What causes depression?",
    "How does exercise affect depression?",
    "What are the signs of a mental health crisis?"
]

# Function to handle question button clicks
def submit_question(question):
    add_message("user", question)
    response = generate_answer(question)
    add_message("bot", response)
    st.experimental_rerun()

# Display questions as buttons in columns
for i, question in enumerate(questions):
    if i % 3 == 0:
        button_col = col1
    elif i % 3 == 1:
        button_col = col2
    else:
        button_col = col3

    if button_col.button(question):
        submit_question(question)

# Proactive assistance based on user interactions
if len(st.session_state.messages) > 3 and "bot_proactive" not in st.session_state:
    proactive_message = "It seems like you have been asking multiple questions. Do you need further assistance or would you like to talk to a human support?"
    add_message("bot", proactive_message)
    st.session_state["bot_proactive"] = True
    st.button("Talk to Human Support", on_click=lambda: add_message("user", "Talk to Human Support"))
