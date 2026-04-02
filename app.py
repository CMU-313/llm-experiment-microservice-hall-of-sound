"""
HTTP entrypoint for the translator microservice.

Downstream apps (other repos / NodeBB, etc.) call this process over HTTP; they are
not Python imports of this codebase. Bind host/port for remote callers; Ollama
runs separately and is configured via OLLAMA_HOST / OLLAMA_MODEL.
"""
import os

from flask import Flask
from flask import request, jsonify
from src.translator import translate_content

app = Flask(__name__)

@app.route("/")
def translator():
    content = request.args.get("content", default = "", type = str)
    is_english, translated_content = translate_content(content)
    return jsonify({
        "is_english": is_english,
        "translated_content": translated_content,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
