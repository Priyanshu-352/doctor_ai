import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from openai import OpenAI

# -------------------------------------------------------------------
# 1. Initialization (Runs on every user interaction)
# -------------------------------------------------------------------
load_dotenv()

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# Qdrant Cloud connection details
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# Create raw Qdrant client (no embedding API calls)
raw_qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Initialize Qdrant Vector Store directly — bypass validation to avoid Gemini quota hit
qdrant = QdrantVectorStore(
    client=raw_qdrant_client,
    collection_name="my_documents",
    embedding=embeddings,
    validate_collection_config=False,
)

# Initialize OpenAI Client (using Gemini API)
SECRET_KEY = os.getenv('GOOGLE_API_KEY')
client = OpenAI(
    api_key=SECRET_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)


st.set_page_config(page_title="My RAG Assistant", page_icon="🤖")
st.title("🤖 My RAG Assistant")
st.write("Ask me questions based on the uploaded documents!")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_query := st.chat_input("Ask a question about your documents..."):

    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:

            with st.spinner("Searching documents..."):
                # Use MMR for diverse, high-quality retrieval (catches deeper concepts)
                results = qdrant.max_marginal_relevance_search(
                    user_query, k=6, fetch_k=20, lambda_mult=0.7
                )
                context = "\n\n".join([
                    f"--- Document {i+1} ---\n{res.page_content}\n[Source: {res.metadata.get('source', 'unknown')}]"
                    for i, res in enumerate(results)
                ])

            SYSTEM_PROMPT = f"""You are an expert AI research assistant. Carefully analyze the provided context to answer the user's question thoroughly and insightfully.

**Context from relevant documents:**
{context}

**Instructions:**
- Synthesize information from ALL the provided documents to give a complete answer
- If the context contains partial information, reason step-by-step to connect concepts
- If the answer cannot be found in the context, say "I couldn't find specific information about this in the documents" — do NOT make up answers
- Use a professional, academic tone
- Cite specific sections from the documents where relevant"""

            with st.spinner("Generating answer..."):
                response = client.chat.completions.create(
                    model="gemini-2.5-flash",
                    n=1,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_query}
                    ]
                )

                answer = response.choices[0].message.content
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"An error occurred: {e}")
