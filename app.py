from flask import Flask, request, jsonify, render_template
from inference_pipeline import chat

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def predict_datapoint():
    user_input = request.json['user_input']
    reply = chat(user_input)
    return jsonify(reply=reply)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
