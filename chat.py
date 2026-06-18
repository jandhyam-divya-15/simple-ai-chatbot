import json
import pickle
import random

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("intents.json") as file:
    intents = json.load(file)

def get_response(message):
    X = vectorizer.transform([message])
    prediction = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == prediction:
            return random.choice(intent["responses"])

    return "Sorry, I don't understand."