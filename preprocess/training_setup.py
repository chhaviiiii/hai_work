import pandas as pd
from torch.utils.data import Dataset
import os
import pickle
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Example of a simple feed-forward neural network model for VTNet
class SimpleVTNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleVTNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class VTNetDataset(Dataset):
    def __init__(self, data_files, data_dir):
        self.data_files = data_files
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if the data is a DataFrame
        if isinstance(data, pd.DataFrame):
            print(f"Loaded DataFrame from {file_path}")
            # Assuming all columns are features, you can adjust if you have specific label columns
            features = data.values  # Convert DataFrame to NumPy array (all columns)
        else:
            print(f"Unexpected data structure in {file_path}: {type(data)}")
            features = data  # If the structure isn't a DataFrame, we assume it's already in the correct format
            labels = None  # Set labels to None if the data structure isn't a tuple

        # Convert features to tensor
        features = torch.tensor(features, dtype=torch.float32)
        
        # If labels exist, process them as well (assuming it's a tuple)
        if isinstance(data, tuple) and len(data) == 2:
            labels = data[1]
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = None  # If labels are not present, set them to None
        
        return features, labels

def train_vtnet_model(output_directory, trained_model_directory, batch_size=32, epochs=10, learning_rate=0.001):
    # Load pickle files from the output directory
    data_files = [f for f in os.listdir(output_directory) if f.endswith('.pkl')]
    
    if not data_files:
        print(f"No .pkl files found in {output_directory}. Please ensure the directory has valid files.")
        return
    
    # Split into train and validation sets
    train_files, val_files = train_test_split(data_files, test_size=0.2, random_state=42)

    # Prepare the datasets and dataloaders
    train_dataset = VTNetDataset(train_files, output_directory)
    val_dataset = VTNetDataset(val_files, output_directory)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_size = 1200  # Adjust to match your feature size
    hidden_size = 128  # You can change this
    output_size = 2    # Adjust based on your number of classes
    model = SimpleVTNet(input_size, hidden_size, output_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        # Calculate the training accuracy
        train_accuracy = correct_preds / total_preds * 100
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct_preds += (predicted == labels).sum().item()
                val_total_preds += labels.size(0)

        # Calculate the validation accuracy
        val_accuracy = val_correct_preds / val_total_preds * 100
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Create the directory for storing the trained model if it doesn't exist
    if not os.path.exists(trained_model_directory):
        os.makedirs(trained_model_directory)

    # Save the model after training
    model_save_path = os.path.join(trained_model_directory, 'trained_vtnet_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Example Usage
output_directory = "/ubc/cs/home/c/cnayyar/hai_work/output"  # Path to your pickle files
trained_model_directory = "/ubc/cs/home/c/cnayyar/hai_work/trained"  # Directory to save the trained model
train_vtnet_model(output_directory, trained_model_directory)
