# Codle

Codle is a full-stack web application mimicing wordle but for algorithm problems.

---

## 🚀 Features

- 🔐 **User Authentication** (Login, Register, Profile)
- 🧠 **Python Backend API** using Flask or similar framework
- 🗄️ **SQLite Database** for local data persistence
- 🌐 **Frontend Interface** built with HTML, CSS, and JavaScript
- 🐳 **Docker Support** for containerized deployment
- ⚙️ **Prettier** configuration for consistent code formatting

---

## 🧩 Project Structure

```
Codle/
├── backend/
│   ├── server.py               # Main backend server (Flask/FastAPI)
│   ├── db_reader.py            # Database read utilities
│   ├── codle.db                # SQLite database
│   ├── requirements.txt        # Python dependencies
│   └── ollama-backend/
│       └── Dockerfile          # Backend container configuration
│
├── frontend/
│   ├── index.html              # Main interface
│   ├── login.html              # Login page
│   ├── register.html           # Register page
│   ├── profile.html            # Profile page
│   ├── style.css               # Main stylesheet
│   ├── auth.css                # Auth-specific styles
│   └── script.js               # Client-side logic
│
├── start_backend.sh            # Script to start the backend server
├── package.json                # Node.js dependencies (for formatting/tools)
├── .prettierrc                 # Prettier configuration
├── .gitignore
└── README.md                   # You are here
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Codle.git
cd Codle
```

### 2. Setup the Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python server.py
```

The backend will start running on `http://localhost:5000`.

### 3. Run the Frontend
Simply open `frontend/index.html` in your browser or serve it using a local server:
```bash
cd frontend
python -m http.server 8080
```
Frontend available at `http://localhost:8080`.

---

## 🐳 Run with Docker (Optional)
To build and run the backend using Docker:
```bash
cd backend/ollama-backend
docker build -t codle-backend .
docker run -p 5000:5000 codle-backend
```

---

## 📦 Dependencies

### Backend (Python)
- Flask / FastAPI (depending on server.py implementation)
- SQLite3
- Other dependencies listed in `backend/requirements.txt`

### Frontend
- Vanilla JS (no framework)
- CSS and HTML
- Prettier (for code formatting)

---

## 🧠 Usage

1. Register a new user on `/register.html`
2. Log in using `/login.html`
3. Access your user profile on `/profile.html`
4. Backend handles database storage and API routes.

---

## 👥 Contributors

- **Charles Prioux**
- **Pierre-Louis Bardon**

---

## 🪪 License

This project is licensed under the **MIT License** – see the LICENSE file for details.