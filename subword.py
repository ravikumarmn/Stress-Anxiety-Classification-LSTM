import nltk
from nltk.tokenize import word_tokenize
import json

nltk.download('punkt')

txt_file = open("dataset/X_vals_google_news_300.txt", "r")
file_content = txt_file.read()

content_list = file_content.split("\n")
txt_file.close()
content_list.remove("")
X=content_list

def parse(d):
    dictionary = dict()
    # Removes curly braces and splits the pairs into a list
    pairs = d.strip('{}').split(', ')
    for i in pairs:
        pair = i.split(': ')
        # Other symbols from the key-value pair should be stripped.
        dictionary[pair[0].strip('\'\'\"\"')] = pair[1].strip('\'\'\"\"')
    return dictionary

def extract_data():
    x_final=[]
    try:
        geeky_file = open('dataset/sub_emotions_output_google_news_300.txt', 'rt')
        lines = geeky_file.read().split('\n')
        for l in lines:
            if l != '':
                dictionary = parse(l)
                x_final.append(dictionary)
        geeky_file.close()
    except:
        print("Something unexpected occurred!")

    x1final = x_final[0]
    x2final = x_final[1]
    x3final = x_final[2]
    x4final = x_final[3]
    x5final = x_final[4]
    x6final = x_final[5]
    x7final = x_final[6]
    x8final = x_final[7]
    x9final = x_final[8]
    x10final=x_final[9]

    x1final = dict([a, int(x)] for a, x in x1final.items())
    x2final = dict([a, int(x)] for a, x in x2final.items())
    x3final = dict([a, int(x)] for a, x in x3final.items())
    x4final = dict([a, int(x)] for a, x in x4final.items())
    x5final = dict([a, int(x)] for a, x in x5final.items())
    x6final = dict([a, int(x)] for a, x in x6final.items())
    x7final = dict([a, int(x)] for a, x in x7final.items())
    x8final = dict([a, int(x)] for a, x in x8final.items())
    x9final = dict([a, int(x)] for a, x in x9final.items())
    x10final= dict([a, int(x)] for a, x in x10final.items())
    return (x1final,x2final,x3final,x4final,x5final,x6final,x7final,x8final,x9final,x10final)


def find_sentence_sub_emotion(a,all_final):
    all_emotions = []
    if a in X:
        for i in all_final[0]:
            if i == a:
                emo = ['anger', all_final[0][i]]
                all_emotions.append(emo)
        for i in all_final[1]:
            if i == a:
                emo = ['anticipation', all_final[1][i]]
                all_emotions.append(emo)
        for i in all_final[2]:
            if i == a:
                emo = ['disgust', all_final[2][i]]
                all_emotions.append(emo)
        for i in all_final[3]:
            if i == a:
                emo = ['fear', all_final[3][i]]
                all_emotions.append(emo)
        for i in all_final[4]:
            if i == a:
                emo = ['joy', all_final[4][i]]
                all_emotions.append(emo)
        for i in all_final[5]:
            if i == a:
                emo = ['negative', all_final[5][i]]
                all_emotions.append(emo)
        for i in all_final[6]:
            if i == a:
                emo = ['positive', all_final[6][i]]
                all_emotions.append(emo)
        for i in all_final[7]:
            if i == a:
                emo = ['sadness', all_final[7][i]]
                all_emotions.append(emo)
        for i in all_final[8]:
            if i == a:
                emo = ['surprise', all_final[8][i]]
                all_emotions.append(emo)
        for i in all_final[9]:
            if i == a:
                emo = ['trust', all_final[9][i]]
                all_emotions.append(emo)
    else:
        return a

    print(a, ' : ', {key: value for key, value in all_emotions})
    
    with open("result/subword_save.json", "r") as f:
        data = json.load(f)
    
    data[a] = {key: value for key, value in all_emotions}
    
    with open("result/subword_save.json", "w") as f:
        json.dump(data, f)
    
    if len(all_emotions) > 1:
        emotion = all_emotions[0][0]
        highest = all_emotions[0][1]
        for i in all_emotions:
            if i[1] > highest:
                emotion = i[0]
                highest = i[1]
        return emotion + str(highest)
    else:
        return all_emotions[0][0] + str(all_emotions[0][1])

