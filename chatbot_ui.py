import streamlit as st
import requests

# FastAPI backend URL
FASTAPI_URL = "http://127.0.0.1:8001/api/question"  # Update if hosted elsewhere

# Set Page Config
st.set_page_config(page_title="Deloitte Chatbot", layout="wide")

# Custom CSS for styling
st.markdown(
   """
<style>

    /* Remove space above title */
    .main > div:first-child {
        padding-top: 0px !important;
        margin-top: 0px !important;
    }

    /* Sidebar content */
    .sidebar-content, .stFileUploader label {
        color: white !important;
    }

    /* Uploaded file name */
    .uploaded-file-name {
        color: black !important;
        font-weight: bold;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        width: fit-content;
        max-width: 80%;
    }
    
    /* User messages */
    .user {
        background-color: #21262d; /* Slightly lighter dark gray */
        color: white;
        align-self: flex-end;
    }

    /* Assistant messages */
    .assistant {
        background-color: #b5d99c; /* Light Green */
        color: black;
        align-self: flex-start;
    }

    /* Tabs Styling */
    div[data-baseweb="tab"] {
        background-color: #007A33 !important; /* Deloitte Green */
        color: white !important; 
        padding: 10px;
        border-radius: 10px;
    }

    /* Active Tab */
    div[data-baseweb="tab"] [aria-selected="true"] {
        background-color: #005a24 !important; /* Darker Green */
        color: white !important;
    }

    /* Tab content */
    .stTabs {
        background-color: #005a24 !important; /* Dark Green */
        border-radius: 10px;
        padding: 1rem;
    }

    /* Ensuring inner tab content is visible */
    .tab-content {
        background-color: #007A33 !important; /* Deloitte Green */
        color: white !important;
        padding: 1rem;
        border-radius: 10px;
    }

    /* Input Box */
    .stTextInput>div>div>input {
        border-radius: 10px !important;
        padding: 10px;
        border: 1px solid #ccc;
    }

    /* Clear Chat Button */
    .stButton>button {
        background: #007A33; /* Deloitte Green */
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background: #004d23; /* Slightly darker green */
    }

</style>

    """,
    unsafe_allow_html=True
)

#Sidebar - Deloitte Logo (Uncomment and add logo path if needed)
st.sidebar.image("logo.png", use_container_width=True)

# st.sidebar.markdown('<h2 style="color: black;">📂 Document Uploader</h2>', unsafe_allow_html=True)
# uploaded_files = st.sidebar.file_uploader("", accept_multiple_files=True)

st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

# Chat History Buttons
dummy_history = [
    "What is Deloitte’s impact on AI?", 
    "How does Deloitte handle data security?", 
    "What are Deloitte’s key AI projects?", 
    "What is Deloitte’s AI governance model?",
    "How does Deloitte use Generative AI?",
    "What is Deloitte’s role in GovTech?",
    "How does Deloitte implement RAG in AI?",
    "What AI solutions does Deloitte offer?",
    "How does Deloitte help government AI?",
    "Explain Deloitte’s AI in consulting.",
]

# Set max length for consistent width
MAX_TITLE_LENGTH = 30

for item in dummy_history:
    display_text = item if len(item) <= MAX_TITLE_LENGTH else item[:MAX_TITLE_LENGTH - 3] + "..."
    st.sidebar.button(display_text, key=item, help=item, use_container_width=True)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Main Chat Layout
st.title("🤖 Deloitte AI Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat Display
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        role_class = "user" if message["role"] == "user" else "assistant"
        st.markdown(
            f'<div class="chat-message {role_class}">{message["content"]}</div>',
            unsafe_allow_html=True,
        )

# User input
user_input = st.chat_input("Ask me anything about Deloitte AI...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        st.markdown(f'<div class="chat-message user">{user_input}</div>', unsafe_allow_html=True)

    # Send request to FastAPI
    try:
        response = requests.post(FASTAPI_URL, json={"question": user_input})
        response_data = response.json()
        answer = response_data.get("answer", "Sorry, I couldn't process that.")
    except requests.exceptions.RequestException:
        answer = "Error connecting to the API. Please try again later."

    # Append and display AI response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with chat_container:
        st.markdown(f'<div class="chat-message assistant">{answer}</div>', unsafe_allow_html=True)

# Clear Chat Button
if st.button("🔄 Clear Chat"):
    st.session_state.messages = []
    st.rerun()
