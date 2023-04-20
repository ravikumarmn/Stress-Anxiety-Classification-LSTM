from flask import Flask, render_template, request
app = Flask(__name__)
import json
from subword import *
import torch

from model import StressClassifier

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
            tokenized_sentence=word_tokenize(text_input)
            final_emotions_list_for_each_word=[]
            
            for i in tokenized_sentence:
                final_emotions_list_for_each_word.append(find_sentence_sub_emotion(i,all_final))
                
            try:   
                input_ids = torch.tensor([word2index[i] for i in text_input.lower().split()],dtype=torch.long)
                probs = model(input_ids.unsqueeze(0))
                
                if probs.item() > 0.5:
                    prediction = params["MAPPING"][1]
                else:
                    prediction = params["MAPPING"][0]
                    
                probability = round(probs.item(), 3)
                
            except KeyError:
                input_text = "I Did not understand."
        
    return render_template("home.html", prediction=prediction, probability=probability, text_input=text_input, input_text=input_text)


if __name__ == "__main__":
    app.run(debug = True)