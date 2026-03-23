import json
import numpy as np
import torch
import torch.nn as nn
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
stemmer = PorterStemmer()


with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = []
    pattern_words = [stemmer.stem(w.lower()) for w in pattern_sentence]
    for w in all_words:
        bag.append(1 if w in pattern_words else 0)

    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)

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

input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000  

model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training shuru ho rahi hai... Please wait.")
for epoch in range(num_epochs):
    words_tensor = torch.from_numpy(X_train)
    labels_tensor = torch.from_numpy(y_train)

    outputs = model(words_tensor)
    loss = criterion(outputs, labels_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "model.pth"
torch.save(data, FILE)
print(f'Training complete! Bot ka dimagh {FILE} mein save ho gaya hai.')