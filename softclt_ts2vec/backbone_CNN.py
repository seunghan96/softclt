import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

train_data = np.load(f'/home/r20user19/Documents/softclt/softclt_ts2vec/datasets/mimic/train_data.npy')
train_label = np.genfromtxt(f'/home/r20user19/Documents/softclt/softclt_ts2vec/datasets/mimic/train_label.csv',delimiter=',')
test_data = np.load(f'/home/r20user19/Documents/softclt/softclt_ts2vec/datasets/mimic/test_data.npy')
test_label = np.genfromtxt(f'/home/r20user19/Documents/softclt/softclt_ts2vec/datasets/mimic/test_label.csv',delimiter=',')

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        print(self.data.shape)
        self.labels = torch.tensor(labels, dtype=torch.long)
        print(self.labels.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataset instances
train_dataset = CustomDataset(train_data, train_label)
test_dataset = CustomDataset(test_data, test_label)

# DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiChannelTimeSeriesCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(MultiChannelTimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def test(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
    
    # Concatenate all batches
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    # Calculate the scores
    probs = torch.softmax(torch.tensor(all_preds), dim=1).numpy()
    #  print(probs)
    preds = np.argmax(all_preds, axis=1)

    # Calculate AUROC and AUPR for the positive class
    auroc = roc_auc_score(all_labels, probs[:, 1])
    aupr = average_precision_score(all_labels, probs[:, 1])
    f1 = f1_score(all_labels, preds)
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss, auroc, aupr, f1

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = MultiChannelTimeSeriesCNN(num_channels=17, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
    test_loss, test_auroc, test_aupr, test_f1 = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}, Test F1 Score: {test_f1:.4f}")

# test_loss, test_auroc, test_aupr, test_f1 = test(model, test_loader, criterion, device)
# print(f"Test Loss: {test_loss:.4f}, Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}, Test F1 Score: {test_f1:.4f}")