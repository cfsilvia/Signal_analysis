import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix


# ==== Dataset Loader divide in windows====
class SignalDataset(Dataset):
    def __init__(self, signal, labels, window_size=256, stride=64):
        self.X = []
        self.Y = []
        for i in range(0, len(signal) - window_size, stride):
            x_win = signal[i:i+window_size]
            y_win = labels[i:i+window_size]
            if not np.isnan(x_win).any():
                self.X.append(x_win)
                self.Y.append(y_win)
        self.X = np.array(self.X, dtype=np.float32)[:, np.newaxis, :]  # shape (N, 1, T)
        self.Y = np.array(self.Y, dtype=np.int64)                       # shape (N, T)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ==== TweetyNet-style 1D Model ====
class TweetyNet1D(nn.Module):
    def __init__(self, input_channels=1, hidden_size=128, n_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))      # (B, 64, T/4)
        x = x.permute(0, 2, 1)                         # (B, T/4, 64)
        x, _ = self.lstm(x)                            # (B, T/4, 2*hidden)
        x = self.classifier(x)                         # (B, T/4, n_classes)
        return x

# ==== Training Function ====
def train(model, dataloader, criterion, optimizer, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)                     # (B, T/4, C)
            y_ds = y[:, ::4]                   # (B, T/4)
            loss = criterion(out.view(-1, out.shape[-1]), y_ds.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

            # collect for metrics
            preds = out.argmax(dim=2)          # (B, T/4)
            all_preds.extend(preds.cpu().numpy().ravel().tolist())
            all_labels.extend(y_ds.cpu().numpy().ravel().tolist())

        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc  = accuracy_score(all_labels, all_preds)
        epoch_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1:02d} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Acc: {epoch_acc:.4f} | "
              f"Prec: {epoch_prec:.4f}")

#==============Prediction==========
def Prediction(model, dataloader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)                           # (B, T/4, C)
            preds = out.argmax(dim=2).cpu().numpy().ravel()
            # downsample labels to match T/4
            y_ds = y[:, ::4].numpy().ravel()
            all_preds.extend(preds)
            all_labels.extend(y_ds)
            
        # 5) Print classification report
        print("=== Classification Report on Validation Set ===")
        print(classification_report(all_labels, all_preds, zero_division=0))

        # 6) Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("=== Confusion Matrix ===")
        print(cm)

#================Inference helper=================================
def predict_downsampled(model, signal, window_size=256, stride=64, device="cpu"):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(signal) - window_size + 1, stride):
            win = signal[start:start+window_size]
            t = torch.from_numpy(win.astype(np.float32))[None, None,:].to(device)
            out = model(t)
            y = out.argmax(dim=-1).squeeze(0)
            preds.append(y.cpu().numpy())
    return np.concatenate(preds, axis = 0)        
        

#====Main Runner train====
def Train_with_tweety(signal, labels,output_file):
    dataset = SignalDataset(signal, labels, window_size=256, stride=64) #create object
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model     = TweetyNet1D(input_channels=1, n_classes=int(labels.max()) + 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train(model, dataloader, criterion, optimizer, device, epochs=100)
    torch.save(model.state_dict(), output_file + "tweetynet1d_model.pth")
    print("Model saved to tweetynet1d_model.pth")
    
#=====Main Runner model with the validation set============
def Validate_with_tweety(signal, labels, model_file, output_file):
    #Build dataset
    dataset = SignalDataset(signal, labels, window_size=256, stride=64) #create object
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    #Instantiate model and load weights
    num_classes = int(labels.max()) + 1
    model = TweetyNet1D(input_channels=1, n_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    # 4) Run inference and collect predictions
    Prediction(model, dataloader, device)
    
#===============Main runner check model with new data=========
def Find_motifs_with_tweety(signal, model_file, output_file):
    #Parameters
    window_size = 256
    stride = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #load saved state and infer n_classes
    state = torch.load(model_file, map_location=device)
    n_classes = state["classifier.weight"].shape[0]
    
    #Build model and load weights
    model = TweetyNet1D(input_channels=1, hidden_size=128, n_classes=n_classes)
    model.load_state_dict(state)
    model.to(device)
    
    #Run inference
    labels_predicted = predict_downsampled(model, signal,window_size=window_size, stride=stride, device=device)
    
    a=1
    