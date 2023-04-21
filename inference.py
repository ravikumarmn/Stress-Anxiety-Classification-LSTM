# import json
# from subword import *
# import torch
# from utils import list2dict
# from model import StressClassifier

# checkpoint_path = "result/stress_detection_using_lstm_emdb_128_hidden_64_epoch_25.pt"
# checkpoint = torch.load(checkpoint_path)

# # # Load the saved train and validation losses and accuracies
# # train_losses = checkpoint['train']['train_losses']
# # train_accuracy = checkpoint['train']['train_accuracy']
# # val_losses = checkpoint['validation']['validation_losses']
# # val_accuracy = checkpoint['validation']['validation_accuracy']


# params = checkpoint["params"]
# vocab_size = checkpoint['vocab_size']

# model = StressClassifier(vocab_size,params)
# model.load_state_dict(checkpoint["model_state_dict"]) # model_state_dict
# model.eval()

# f = json.load(open("result/word_to_idx_to_word.json","r"))
# word2index = f['char_to_idx']
# index2word = f['idx_to_char']
# all_final = extract_data()

# text_input  = "i am stress"
# tokenized_sentence=word_tokenize(text_input)
# emotion_dict =dict()
# for i in tokenized_sentence:
#     emotion_dict[i] = dict(find_sentence_sub_emotion(i,all_final))
#     try:   
#         if text_input in ["q","quit","bye","close"]:
#             break
#         else:
#             input_ids = torch.tensor([word2index[i] for i in text_input.lower().split()],dtype=torch.long)
#             probs = model(input_ids.unsqueeze(0))
#             if probs.item() > 0.5:
#                 out = params["MAPPING"][1]
#                 print({out:round(probs.item(),2)})
#             else:
#                 out = params["MAPPING"][0]
#                 print({out:round(probs.item(),2)})
#     except KeyError:
#         print("Given input is not valid")
