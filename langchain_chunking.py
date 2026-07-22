# -------------------------------------------------------------------
# PDF Ingestion Script
# Loads PDF, chunks it (larger chunks for deeper concepts),
# embeds with Gemini, stores in Qdrant Cloud.
# Supports resume if daily Gemini quota is hit.
# -------------------------------------------------------------------

from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader

file_path = "./doctor.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages from PDF")

# Larger chunk size to capture deeper concepts and full context
# 1200 chars ≈ 1-2 paragraphs — much better for semantic understanding
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,       # Increased from 500 — captures full sections
    chunk_overlap=200,      # 17% overlap — enough for coherence
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(docs)
print(f"Split into {len(texts)} chunks (chunk_size=1200)")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.exceptions import ResponseHandlingException
import time
import json

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# Resume support — Gemini free tier allows only 1000 embeddings/day
PROGRESS_FILE = "ingestion_progress.json"
start_idx = 0
try:
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
        start_idx = progress.get("last_index", 0)
        print(f"🔄 Resuming from chunk index {start_idx}")
except (FileNotFoundError, json.JSONDecodeError):
    print("🆕 Starting fresh ingestion")

BATCH_SIZE = 10  # Conservative batch for Qdrant Cloud free tier
MAX_RETRIES = 5
RETRY_DELAY = 10
qdrant = None
embeddings_count = 0

# Try to connect to existing collection first
try:
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="my_documents",
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    print("📦 Found existing collection 'my_documents', appending...")
except Exception as e:
    if "not found" in str(e).lower() or "404" in str(e):
        print("📦 Creating new collection 'my_documents'...")
    else:
        print(f"⚠️ Connection issue: {e}")
        print("📦 Will try to create collection on first batch...")

for i in range(start_idx, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    batch_num = i//BATCH_SIZE + 1
    total_batches = (len(texts)-1)//BATCH_SIZE + 1
    print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks, index {i})...")
    
    success = False
    for attempt in range(MAX_RETRIES):
        try:
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
            
            # Save progress after each successful batch
            with open(PROGRESS_FILE, "w") as f:
                json.dump({"last_index": i + len(batch)}, f)
            embeddings_count += len(batch)
            success = True
            break
        except (ResponseHandlingException, Exception) as e:
            error_str = str(e)
            print(f"  ⚠️ Attempt {attempt+1} failed: {error_str[:120]}...")
            
            # Check for Gemini daily quota exhaustion
            if "RESOURCE_EXHAUSTED" in error_str:
                print(f"\n⛔ Gemini free tier quota reached ({embeddings_count} embeddings used today).")
                print("   Resume tomorrow by running: python langchain_chunking.py")
                print(f"   Progress saved at index {i}. It will auto-resume.\n")
                exit(0)
            
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    
    if not success:
        print(f"  ❌ Failed after {MAX_RETRIES} attempts, skipping batch")
    
    # Small delay between batches
    if i + BATCH_SIZE < len(texts):
        time.sleep(2)

# Clean up progress file on completion
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)

print(f"\n✅ Ingestion complete! {embeddings_count} chunks stored in Qdrant Cloud.")
print(f"   Now run: streamlit run rag_app.py")
