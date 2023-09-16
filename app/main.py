from flask import Flask
from flask_cors import CORS
from interact import Chatbot

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

chatbot = Chatbot()
chatbot.run()


@app.route("/")
def index():
    return "No mask L"


@app.route("/message/<user_input>")
def send_message(user_input):
    user_input = user_input.replace("_", " ")
    user_input = user_input.lower()
    return chatbot.predict(user_input)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
