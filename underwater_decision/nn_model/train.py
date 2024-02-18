import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.optim import Adam

from model import MarineFoulingNet
from dataset import MarineFoulingDataset


def train(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(inputs), labels) for inputs, labels in val_loader) / len(val_loader)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss}')


if __name__=="__main__":
    df = pd.read_csv('/workspaces/UnderWater-Decision/data/default_synthetic_dataset1.csv')
    X = df.drop('Recommended_Cleaning_Method', axis=1)
    y = LabelEncoder().fit_transform(df['Recommended_Cleaning_Method'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = MarineFoulingDataset(X_train, y_train)
    val_dataset = MarineFoulingDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize Model
    input_size = X.shape[1]
    hidden_size = 100
    num_classes = len(set(y))
    model = MarineFoulingNet(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Train Model
    train(model, criterion, optimizer, train_loader, val_loader)
