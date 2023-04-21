from flask import Flask, render_template, request

app = Flask(__name__)
import json
import os

import torch

from model import StressClassifier
from subword import *

checkpoint_path = "result/stress_detection_using_lstm_emdb_128_hidden_64_epoch_25.pt"
checkpoint = torch.load(checkpoint_path)

params = checkpoint["params"]
vocab_size = checkpoint['vocab_size']

model = StressClassifier(vocab_size,params)
model.load_state_dict(checkpoint["model_state_dict"]) # model_state_dict
model.eval()

f = json.load(open("result/word_to_idx_to_word.json","r"))
word2index = f['char_to_idx']
index2word = f['idx_to_char']
all_final = extract_data()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    text_input = None
    input_text = None
    
    if request.method == "POST":
        text_input = request.form.get('text')
        
        if text_input:
            empty_dict = {}
            with open("result/subword_save.json", "w") as f:
                json.dump(empty_dict, f)

            tokenized_sentence=word_tokenize(text_input)
            my_dict_final = dict()
            for i in tokenized_sentence:
                my_dict_final[i] = find_sentence_sub_emotion(i, all_final)

            try:   
                input_ids = torch.tensor([word2index[i] for i in text_input.lower().split()],dtype=torch.long)
                probs = model(input_ids.unsqueeze(0))
                
                if probs.item() > 0.5:
                    prediction = params["MAPPING"][1]
                else:
                    prediction = params["MAPPING"][0]
                    
                probability = round(probs.item(), 3)
                
            except KeyError:
                input_text = "I Did Not Understand."
        
    return render_template("home.html", prediction=prediction, probability=probability, text_input=text_input, input_text=input_text)

@app.route("/result")
def result():
    images = ["static/acc_conf_mat.png"]
    return render_template("result.html", images=images)


if __name__ == "__main__":
    app.run(debug = True)