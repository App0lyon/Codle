#!/bin/bash
set -e

# --- Configuration ---
ROOT_DIR="$(dirname "$0")"
BACKEND_DIR="$ROOT_DIR/backend"
VENV_DIR="$BACKEND_DIR/venv"
PORT=8000

echo "Lancement du backend depuis : $ROOT_DIR"

# --- Vérification du dossier backend ---
if [ ! -d "$BACKEND_DIR" ]; then
    echo "Le dossier backend/ est introuvable. Exécute ce script depuis la racine du projet."
    exit 1
fi

# --- Création de l'environnement virtuel si nécessaire ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv "$VENV_DIR"
fi

# --- Activation de l'environnement virtuel ---
echo "Activation de l'environnement virtuel..."
source "$VENV_DIR/bin/activate"

# --- Installation des dépendances ---
REQ_FILE="$BACKEND_DIR/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "Installation des dépendances..."
    pip install --upgrade pip > /dev/null
    pip install -r "$REQ_FILE"
else
    echo "Fichier requirements.txt introuvable dans $BACKEND_DIR"
fi

# --- Variables optionnelles pour Ollama ---
export OLLAMA_HOST=${OLLAMA_HOST:-"http://localhost:11434"}
export OLLAMA_MODEL=${OLLAMA_MODEL:-"gemma3:4b"}

echo "OLLAMA_HOST = $OLLAMA_HOST"
echo "OLLAMA_MODEL = $OLLAMA_MODEL"

# --- Libération du port si déjà utilisé ---
if sudo lsof -i :$PORT >/dev/null 2>&1; then
    echo "Le port $PORT est déjà utilisé, arrêt du processus existant..."
    sudo kill $(sudo lsof -t -i:$PORT) 2>/dev/null || true
fi

# --- Lancement du serveur FastAPI ---
echo "Démarrage du backend sur le port $PORT..."
cd "$BACKEND_DIR"
uvicorn server:app --host 0.0.0.0 --port $PORT

