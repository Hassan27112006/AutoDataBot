from flask import Flask, render_template, request, jsonify, send_file
import os, uuid, datetime
from core.automl_engine import AutoMLEngine
from core.chatbot_engine import ChatbotEngine
from core.memory_manager import MemoryManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE_DIR, "uploads")
PROJECTS = os.path.join(BASE_DIR, "projects")
INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")

os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(PROJECTS, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB upload cap

# Shared managers/engines
memory = MemoryManager(os.path.join(BASE_DIR, "memory_store.db"))

# -----------------------------
# Initialize Chatbot Engine
# -----------------------------
chat_engine = ChatbotEngine(
    dataset_index_dir=os.path.join(BASE_DIR, "data", "faiss_index"),  # your actual folder
    general_index_dir=os.path.join(BASE_DIR, "data", "faiss_general")
)

# Auto-load FAISS index if exists
if os.path.exists(INDEX_DIR):
    try:
        chat_engine.load_index(INDEX_DIR)
        print("📌 FAISS index loaded successfully.")
    except Exception as e:
        print("⚠️ Could not load FAISS index:", e)
else:
    print("⚠️ No FAISS index found. Chatbot will say 'No relevant knowledge found'.")

auto_engine = AutoMLEngine(memory=memory, projects_dir=PROJECTS)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    message = payload.get("message", "")

    # Chatbot response
    reply = chat_engine.answer(message, top_k=5)

    return jsonify({"response": reply})


@app.route("/analyze", methods=["POST"])
def analyze():
    from werkzeug.utils import secure_filename

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    path = os.path.join(UPLOADS, filename)
    file.save(path)

    target = request.form.get("target") or None

    try:
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        project_name = f"project_{timestamp}_{uuid.uuid4().hex[:6]}"
        project_path, summary = auto_engine.run_pipeline(path, target, project_name)

        zip_path = auto_engine.package_project(project_path)
        memory.add_run(summary)

        return jsonify({
            "message": "Analysis complete",
            "zip": os.path.basename(zip_path),
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<zipname>", methods=["GET"])
def download(zipname):
    p = os.path.join(PROJECTS, zipname)
    if os.path.exists(p):
        return send_file(p, as_attachment=True)
    return jsonify({"error": "Not found"}), 404


def secure_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "._- ").strip()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
