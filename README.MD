# From project root
docker build -t codle-ollama -f backend/ollama-backend/Dockerfile .
docker run -d --name codle-ollama -p 11434:11434 codle-ollama

cd backend/
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional: point backend at the containerized Ollama
export OLLAMA_HOST="http://localhost:11434"
# Optional: pick a model tag if you use a different one than in the Dockerfile
export OLLAMA_MODEL="gemma3:4b"

uvicorn server:app --reload --port 8000



---
GET /health should respond with { ok: true, model, ollama }.

curl -X POST \
  http://localhost:8000/problems \
  -H "Content-Type: application/json" \
  -d '{"difficulty":"medium"}'
