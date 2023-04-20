import torch.nn as nn

class StressClassifier(nn.Module):
    def __init__(self, vocab_size,params):
        super(StressClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, params["EMBEDDING_DIM"])
        self.lstm = nn.LSTM(params["EMBEDDING_DIM"], params["HIDDEN_DIM"])
        self.fc = nn.Linear(params["HIDDEN_DIM"], params["OUTPUT_FEATURES"])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (_, _) = self.lstm(embedded)
        logits = self.fc(output[:, -1, :])
        prediction = self.sigmoid(logits)
        return prediction