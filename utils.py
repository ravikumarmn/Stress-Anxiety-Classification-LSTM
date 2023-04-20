import re

import pandas as pd
import torch
from sklearn.metrics import accuracy_score


def read_csv_file(params,dataset_type = "train"):
    data_file_str = params["DATA_DIR"] + dataset_type+".csv"
    df = pd.read_csv(
        data_file_str,
        usecols=["subreddit","text","id","label"]
    )
    df = preprocess_text(df,'text')
    return df


def preprocess_text(df, column):
    df[column] = df[column].apply(lambda x: re.sub(r'\W+', ' ', x))
    df[column] = df[column].apply(lambda x: x.lower())
    # Remove extra whitespaces
    df[column] = df[column].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    return df

def train_model(model, train_loader, val_loader, optimizer, criterion,params,vocab_size):
    best_val_loss = float('inf')
    counter = 0
    train_losses = list()
    train_accuracy = list()
    val_losses = list()
    val_accuracy = list()
    for epoch in range(params["EPOCHS"]):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids']
            labels = batch['label']
            optimizer.zero_grad()
            probs = model(input_ids)
            loss = criterion(probs.flatten(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += accuracy_score(labels.detach().numpy(), probs.detach().numpy().round())
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                labels = batch['label']
                probs = model(input_ids)
                loss = criterion(probs.flatten(), labels)
                val_loss += loss.item()
                val_acc += accuracy_score(labels.detach().numpy(), probs.detach().numpy().round())
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy.append(val_acc)
        print(f'Epoch {epoch+1}/{params["EPOCHS"]} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            save_dict = {
                "model_state_dict":model.state_dict(),
                "params":params,
                "train" : {
                    "train_losses" : train_losses,
                    "train_accuracy" : train_accuracy
                },
                "validation" : {
                    "validation_losses" : val_losses,
                    "validation_accuracy" : val_accuracy
                },
                "epoch" : f"{epoch}",
                "vocab_size":vocab_size
                
            }
            torch.save(save_dict, params["MODEL_SAVE_PATH"])
        else:
            counter += 1
            if counter >= params["PATIENCE"]:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    print('Training complete!')

