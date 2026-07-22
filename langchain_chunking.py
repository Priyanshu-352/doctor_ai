# -------------------------------------------------------------------
# PDF Ingestion Script
# Loads PDF, chunks it, embeds with Gemini, stores in Qdrant Cloud
# -------------------------------------------------------------------

from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader

file_path = "./Rules-Regulations20251105.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages from PDF")

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=250,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(docs)
print(f"Split into {len(texts)} chunks")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

from langchain_qdrant import QdrantVectorStore
import time

# Qdrant Cloud connection details
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# Process in small batches to avoid free-tier API quota limits (100 req/min)
BATCH_SIZE = 20
qdrant = None

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(texts)-1)//BATCH_SIZE + 1} ({len(batch)} chunks)...")
    
    if qdrant is None:
        qdrant = QdrantVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name="my_documents",
        )
    else:
        qdrant.add_documents(batch)
    
    # Sleep to respect free-tier rate limits (max ~100 requests/min)
    if i + BATCH_SIZE < len(texts):
        time.sleep(15)

print(f"✅ {len(texts)} chunks stored in Qdrant Cloud at '{QDRANT_URL}'")
