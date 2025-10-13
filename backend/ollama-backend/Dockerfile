FROM ollama/ollama:latest

ENV OLLAMA_HOST 0.0.0.0:11434

ENV OLLAMA_MODELS /models

ENV OLLAMA_DEBUG false

ENV OLLAMA_KEEP_ALIVE -1

ENV MODEL gemma3:4b
RUN ollama serve & sleep 5 && ollama pull $MODEL

ENTRYPOINT ["ollama", "serve"]

EXPOSE 11434