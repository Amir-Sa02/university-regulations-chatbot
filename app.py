from flask import Flask, render_template, request, jsonify
from rag_core import answer_with_rag, memory

app = Flask(__name__)

@app.route("/")
def index():
    """Serves the main HTML page and resets the chat memory for a new session."""
    # We now check for and reset the 'memory' object directly.
    if memory:
        memory.reset() 
        print("Chat history has been reset for the new session.")
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles the chat message from the user."""
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    # The rag_core function now handles its own memory.
    bot_response = answer_with_rag(user_message)
    
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
