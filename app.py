from flask import Flask, render_template, request, jsonify
# We now need access to the chat_engine object to reset it.
from rag_core import answer_with_rag, chat_engine

app = Flask(__name__)

@app.route("/")
def index():
    """Serves the main HTML page and resets the chat engine's memory for a new session."""
    if chat_engine:
        chat_engine.reset() # This clears the memory for a fresh start.
        print("Chat history has been reset for the new session.")
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles the chat message from the user."""
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    # We no longer need to pass the history. The engine manages it internally.
    bot_response = answer_with_rag(user_message)
    
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
