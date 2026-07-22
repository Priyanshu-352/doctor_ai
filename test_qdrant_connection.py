"""
Quick test script to verify Qdrant Cloud connection and list collections.
Run this first to make sure your Qdrant Cloud credentials work.
"""
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

print(f"🔗 Connecting to Qdrant Cloud at: {QDRANT_URL}")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

try:
    # Try to access existing collection (may fail if not yet created — that's okay)
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="my_documents",
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    print("✅ Connection successful! Collection 'my_documents' exists.")
    print("   You can now run: streamlit run rag_app.py")
    
except Exception as e:
    error_msg = str(e)
    if "not found" in error_msg.lower() or "404" in error_msg:
        print("✅ Connection successful! (Collection 'my_documents' not yet created)")
        print("   Run this to create & populate it:")
        print("   python langchain_chunking.py")
    else:
        print(f"❌ Connection failed: {error_msg}")
        print("")
        print("📋 Troubleshooting:")
        print("   1. Check your .env file has QDRANT_URL and QDRANT_API_KEY set correctly")
        print("   2. Make sure the URL ends with :6333")
        print("   3. Verify the API key in your Qdrant Cloud dashboard")

