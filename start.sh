#!/bin/bash
# Start FastAPI backend in the background. Bound to 0.0.0.0 to allow access from host browser (Swagger UI).
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend in the foreground. Exposed to 0.0.0.0 for external access.
streamlit run src/frontend/app.py --server.port=8501 --server.address=0.0.0.0
