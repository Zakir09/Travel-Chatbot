from flask import Flask, request, jsonify
from flask_cors import CORS
import chatbot  # Import your Python script

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.json['userInput']
    response = chatbot.chat_gpt(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

