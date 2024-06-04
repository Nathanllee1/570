import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 19200
        
        # Layer 1
        # input: [1, 1, 1, 64]
        self.conv1 = nn.Conv2d(1, 64, (1, 4), padding = 0) # [1, 16, 1, 61]
        self.batchnorm1 = nn.BatchNorm2d(64, False) # [1, 16, 1, 61]
         
        # Layer 2
        # [1, 61, 16, 1]
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(64, 16, (1, 4)) # [1, 4, 1, 63]
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.MaxPool2d(1, 1)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(16, 8, (1, 2))
        self.batchnorm3 = nn.BatchNorm2d(8, False) # [1, 4, 8, 25]
        self.pooling3 = nn.MaxPool2d((1, 1)) # [1, 4, 4, 6]
        
        # FC Layer
        self.fc1 = nn.Linear(6696, 1)

        # final conv layer for 3 outputs
        self.output_layer = nn.Linear(1200, 3)


        

    def forward(self, x):
        # Layer 1
        # print("Input X: ", x.shape)
        x = F.relu(self.conv1(x))
        # print("After first conv: ", x.shape)
        x = self.batchnorm1(x)
        # print("after batchnorm1", x.shape)
        x = F.dropout(x, 0.25)
        # print("after dropout: ", x.shape)
        # x = x.permute(0, 3, 1, 2)
        # print("after dropout and permute: ", x.shape)
        
        # Layer 2
        x = self.padding1(x)
        x = F.relu(self.conv2(x))
        # print("After second conv: ", x.shape)
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)
        # print("post batchnorm3: ", x.shape)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # print("post layer 3: ", x.shape)
        
        # x = self.output_layer(x)
        # x = F.softmax(x, dim=1)
        # # FC Layer
        x = x.reshape(-1, 6696)
        # print("post FC: ", x.shape)
        x = self.fc1(x)
        # print("pre sigmoid x: ", x.shape)
        # x = F.softmax(x, dim=1)
        x = F.sigmoid(x)
        # x = F.sigmoid(self.fc1(x))
        # print("final x?: ", x.shape)
        return x
    
class SimpleEEGNet(nn.Module):
    def __init__(self):
        super(SimpleEEGNet, self).__init__()
        self.lin1 = nn.Linear(64, 100)
        # relu
        self.lin2 = nn.Linear(100, 50)
        # relu
        self.lin3 = nn.Linear(50, 1)
        # sigmoid
    
    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.sigmoid(x)

        return x

def run_simple_eeg_model(train_loader):
    net = SimpleEEGNet()
    print(net)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs).view(-1)
            # results = outputBinaryClass(outputs)
            # print(results)
            # print("labels: ", labels)
            # print("output shape", outputs.shape)
            # print("labels shape, ", labels.shape)
            loss = criterion(outputs, labels)
            # loss.requires_grad = True
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # if(epoch == num_epochs-1):
            #     print(results)
            #     print("labels: ", labels)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

def run_eeg_model(train_loader, val_loader, test_loader, train=True, model_path='t1t2_eegnet.pth', log_dir='runs/eegnet'):
    net = EEGNet()
    print(net)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    writer = SummaryWriter(log_dir)

    # sample_tensor = torch.unsqueeze(X_train_tensor[0], 1)
    # print(sample_tensor.shape)
    # outs = net(sample_tensor)


    # Training loop
    if(train):
        print("Training the model")
        num_epochs = 30
        for epoch in range(num_epochs):
            net.train()
            running_acc = 0.0
            running_loss = 0.0
            for inputs, labels in train_loader:
                labels = torch.unsqueeze(labels, 1)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = (outputs.round() == labels).float().mean()
                running_acc += acc

                running_loss += loss.item()
                # print("accuracy: ", acc, "loss: ", loss)

                # if(epoch == num_epochs-1):
                #     print(results)
                #     print("labels: ", labels)

            avg_loss = running_loss / len(train_loader)
            avg_acc = running_acc / len(train_loader)
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Accuracy/train', avg_acc, epoch)

            # Validation
            net.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    labels = torch.unsqueeze(labels, 1)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    acc = (outputs.round() == labels).float().mean()
                    val_acc += acc

            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_acc / len(val_loader)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/val', avg_val_acc, epoch)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss}, Train Acc: {avg_acc}, Val Loss: {avg_val_loss}, Val Acc: {avg_val_acc}')
            
            # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Acc: {running_acc/len(train_loader)}')

        # Save the trained model
        torch.save(net.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print(f"Loading model from {model_path}")
        net.load_state_dict(torch.load(model_path))
    
    # Testing loop
    net.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = torch.unsqueeze(labels, 1)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            predictions = outputs.round()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Store all predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct_predictions / total_samples * 100
    writer.add_scalar('Loss/test', avg_test_loss, 0)
    writer.add_scalar('Accuracy/test', accuracy, 0)

    print(f'Test Loss: {avg_test_loss}, Accuracy: {accuracy}%')

    # Calculate confusion matrix
    all_labels = np.array(all_labels).flatten()
    all_predictions = np.array(all_predictions).flatten()
    cm = confusion_matrix(all_labels, all_predictions)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(cm)

    writer.close()

def main():
    # sample_data = pd.read_csv('data.csv')
    # X = sample_data[sample_data.columns[3:]]
    # y = sample_data[sample_data.columns[2]]

    full_data = np.load('data/cleanedData.npy', allow_pickle=True)
    real_data = np.load('data/real_data.npy')
    imagined_data = np.load('data/imagined_data.npy')
    full_data = full_data

    # Create a mask for rows where the label is not 'T0'
    mask = full_data[:, 1] != 'T0'
    # Apply the mask to full_data to keep only 'T1' and 'T2'
    filtered_data = full_data[mask]
    full_data = filtered_data

    print(full_data)
    X = full_data[:, 2:]
    y = full_data[:, 1]
    print(y)

    # label_mapping = {'T0': 0, 'T1': 1, 'T2': 1}
    label_mapping = {'T1': 0, 'T2': 1}
    y = np.array([label_mapping[label] for label in y])

    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # Get the unique classes and their counts
    classes, counts = np.unique(y_train, return_counts=True)

    # Print the class counts
    for cls, count in zip(classes, counts):
        print(f"Class {int(cls)}: {count} samples")

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    print("training shape: ", X_train_tensor.shape)
    print("testing shape: ", X_test_tensor.shape)

    # Create DataLoader for training and testing sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_flag = True
    model_path = "" # defaults to eegnet.pth
    run_eeg_model(train_loader, val_loader, test_loader, train=train_flag)
    # run_simple_eeg_model(train_loader)

def outputBinaryClass(outputs):
    outputs = torch.squeeze(outputs)
    threshold = 0.5
    predictions = (outputs > threshold).float()

    # print(outputs)
    # print(predictions)
    return predictions
def outputClass(tensors):

    results = []

    for tensor in tensors:
        if tensor[0] > tensor[1]:
            results.append(0)
            continue

        results.append(1)

    return torch.tensor(results, dtype=torch.float32)
    

if __name__ == "__main__":
    main()