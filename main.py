from dotenv import load_dotenv
import os
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

# Load environment variables from .env
load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Put it in .env or export it in your environment.")

# Configure the genai client (keeps API key only on the server)
genai.configure(api_key=GEMINI_API_KEY)

# Create a model instance (same pattern you used)
model = genai.GenerativeModel("gemini-2.5-pro")

app = Flask(__name__)

# Simple homepage with a form
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Form POST endpoint (server-rendered result)
@app.route("/ask", methods=["POST"])
def ask():
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return render_template("index.html", error="Please enter a prompt.", prompt=prompt)

    try:
        # synchronous request to Gemini
        response = model.generate_content(prompt)

        # The library response shape can vary; try common fields robustly:
        answer = None
        if hasattr(response, "text"):
            answer = response.text
        elif hasattr(response, "candidates") and len(response.candidates) > 0:
            # some client libs return candidates[0].content
            answer = getattr(response.candidates[0], "content", None)
        else:
            # fallback to stringification for debugging
            answer = str(response)

        return render_template("index.html", prompt=prompt, answer=answer)

    except Exception as e:
        # show a user-friendly error; in prod log this instead
        return render_template("index.html", error=f"API error: {e}", prompt=prompt)

# Lightweight JSON API for AJAX
@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        response = model.generate_content(prompt)
        answer = getattr(response, "text", None) or (
            response.candidates[0].content if hasattr(response, "candidates") else str(response)
        )
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # debug=True is convenient while learning; turn it off in production.
    app.run(host="127.0.0.1", port=5000, debug=True)
