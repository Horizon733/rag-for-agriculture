import streamlit as st
import spacy
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from utils import get_weather  # Ensure utils.py contains a get_weather function.
import requests

# Load spaCy model for entity extraction
nlp = spacy.load("en_core_web_sm")

# Constants
PROMPT_TEMPLATE = """
You are good and intelligent assistant who can answer questions about Agriculture.
You can also provide weather updates for cities.
If the user greets, respond with a greeting. If the user asks a question, provide an answer.
You can use following context too for answering questions:

{context}

Conversation History: 
{history}

---


Answer the question based on the above context: {query}

"""

CHROMA_DB_DIR = "chroma"
OLLAMA_API_URL = "http://localhost:11434"  # Replace with your Ollama server API URL

# Initialize ChromaDB and Embeddings
def initialize_chromadb():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    return vector_store

# Extract City Name from User Input
def extract_city(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":  # Geopolitical Entity (e.g., city, country)
            return ent.text
    return None

st.set_page_config(page_title="Agriculture Chat Assistant", layout="wide")
st.title("🌾 Agriculture Chat Assistant")
st.write("Ask questions about agriculture tips, tricks, or get weather updates!")

# Initialize ChromaDB and LangChain LLM
vector_store = initialize_chromadb()
llm = Ollama(
    base_url=OLLAMA_API_URL, 
    model="llama3.1",  # Replace with the specific model name you are using
    timeout=10  # Add a timeout of 10 seconds
)

# Memory for conversation history
memory = ConversationBufferMemory(
    memory_key="history", 
    return_messages=True,
    input_key="query",
)

# Prompt Template for LangChain
prompt_template = PromptTemplate(
    input_variables=["context", "history", "query"],
    template=PROMPT_TEMPLATE
)

# LangChain LLM Chain
chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)


st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
model_type = st.sidebar.selectbox("Select Model Type", ("Model A", "Model B", "Model C"))
agent_bot = st.sidebar.selectbox("Select Agent Bot", ("Bot 1", "Bot 2", "Bot 3"))

# Chatbox implementation
st.subheader("Chatbox")


# Container for chat messages
chat_container = st.container()

# Function to display chat messages
def display_message(message, is_user=True):
    if is_user:
        chat_container.markdown(f"<div style='text-align: right; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>", unsafe_allow_html=True)
    else:
        chat_container.markdown(f"<div style='text-align: left; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>", unsafe_allow_html=True)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
with chat_container:
    for chat in st.session_state.messages:
        display_message(chat['content'], is_user=chat['is_user'])


# User Input Section
user_input = st.text_input("Enter your query:", key="user_input")
send_button = st.button("Send")

if send_button:
    user_input = st.session_state.user_input.strip()  # Ensure the input is not empty or just whitespace
    weather_info = ""

    if user_input:
        # Extract city name for weather info
        city = extract_city(user_input)
        if city:
            try:
                weather_info = get_weather(city)
                st.info(f"Weather update for **{city}**: {weather_info}")
            except Exception as e:
                st.error(f"Error fetching weather for {city}: {e}")

        # Retrieve relevant context from ChromaDB
        try:
            search_results = vector_store.similarity_search(user_input, k=3)
            context = f"{weather_info}\n\n" if weather_info else ""
            for result in search_results:
                context += result.page_content + "\n\n"
        except Exception as e:
            st.error(f"Error retrieving context from ChromaDB: {e}")
            context = ""

        # Generate response using LangChain
        try:
            print("Context:", context)
            response = chain.run(context=context, query=user_input)
            # st.success(response)

            # Update conversation memory
            st.session_state.messages.append({"role": "user", "content": user_input, "is_user":True})
            st.session_state.messages.append({"role": "assistant", "content": response, "is_user":False})
        except requests.exceptions.Timeout:
            st.error("The request to the LLM timed out. Please try again.")
        except Exception as e:
            st.error(f"Error generating response: {e}")
        st.rerun()
    else:
        st.warning("Please enter a valid query.")
# Display Conversation History with Chat UI
# st.write("### Conversation History")

# # Create two columns for chat display
# user_col, assistant_col = st.columns([1, 5])

# # Display user messages and assistant responses
# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         with user_col:
#             st.markdown(f"**You:** {msg['content']}")
#     else:
#         with assistant_col:
#             st.markdown(f"**Assistant:** {msg['content']}")
