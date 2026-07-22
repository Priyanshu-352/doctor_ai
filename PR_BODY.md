## Changes Made

### Qdrant Cloud Connection

- Updated `rag_app.py` to use Qdrant Cloud (`url` + `api_key`) instead of local on-disk storage
- Updated `langchain_chunking.py` to ingest documents to Qdrant Cloud
- Added `test_qdrant_connection.py` to verify cloud connection and troubleshoot

### Deployment Support

- Added `requirements.txt` with all dependencies for cloud deployment
- Updated `.gitignore` to exclude secrets and local storage
- Added environment variables template in `sample.env`

### How to Deploy on Streamlit Community Cloud (Free)

1. Go to https://share.streamlit.io and sign in with GitHub
2. Click "Deploy an app" -> select the `doctor_ai` repository
3. Set main file path to `rag_app.py`
4. In the "Secrets" section, add:
   - `GOOGLE_API_KEY` = your Gemini API key
   - `QDRANT_URL` = your Qdrant Cloud URL (ending in :6333)
   - `QDRANT_API_KEY` = your Qdrant Cloud API key
5. Click Deploy and your app will be live!
