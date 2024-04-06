from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import chatbot  # Import your Python script

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Chatbot page route
@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

# Existing routes for processing input and resetting
@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.json['userInput']
    response = chatbot.chat_gpt(user_input)
    return jsonify({"response": response})

@app.route('/reset', methods=['GET'])
def reset():
    chatbot.travel_plan_state.reset()
    return jsonify({"status": "success", "message": "State reset successfully."})

if __name__ == "__main__":
    app.run(debug=True)

