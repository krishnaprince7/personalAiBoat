from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import json
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import subprocess


nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)
CORS(app)  

stemmer = PorterStemmer()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

def load_brain():
    with open('intents.json', 'r', encoding='utf-8') as json_data:
        intents = json.load(json_data)

    FILE = "model.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()
    return intents, all_words, tags, model

intents, all_words, tags, model = load_brain()

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stemmer.stem(word.lower()) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


@app.route('/chat', methods=['POST'])
def chat():
    
    user_msg = request.json.get("message")
    
    if not user_msg:
        return jsonify({"reply": "Message khali hai!"})

  
    if user_msg.lower().strip().startswith("sikho"):
        try:
            learning_data = user_msg.split(":", 1)[1] 
            sawal, jawab = learning_data.split("|")
            sawal = sawal.strip()
            jawab = jawab.strip()

            with open('intents.json', 'r', encoding='utf-8') as f:
                json_file_data = json.load(f)
            
            new_tag = "learned_" + str(random.randint(1000, 9999))
            json_file_data["intents"].append({"tag": new_tag, "patterns": [sawal], "responses": [jawab]})
            
            with open('intents.json', 'w', encoding='utf-8') as f:
                json.dump(json_file_data, f, indent=2, ensure_ascii=False)
            
            # API background me train karegi
            subprocess.run(["python", "train.py"])
            global intents, all_words, tags, model
            intents, all_words, tags, model = load_brain()
            
            return jsonify({"reply": "Data save aur train ho gaya! Naya dimagh active hai 🧠✨"})
            
        except Exception as e:
            return jsonify({"reply": "Oops! Format galat hai. Aise likhein -> Sikho: sawal | jawab"})

    # Normal Chat Logic
    sentence_tokenized = nltk.word_tokenize(user_msg)
    X = bag_of_words(sentence_tokenized, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return jsonify({"reply": random.choice(intent['responses'])})
    else:
        return jsonify({"reply": "Sorry, mere paas abhi iska jawab nahi hai. Aap mujhe sikhayein 'Sikho: sawal | jawab' likh kar!"})

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)