import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Simple dataset
data = torch.randn(100, 2)  # 100 samples, 2 features
labels = (data[:, 0] + data[:, 1] > 0).float().view(-1, 1)  # Label is 1 if sum of features > 0, else 0

# Normalize the data
data = (data - data.mean(dim=0)) / data.std(dim=0)

# More complex neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Define a function to perform modal evasion attack
def modal_evasion_attack(model, data, labels, epsilon=0.1):
    # Enable gradients for the input data
    data.requires_grad = True
    
    # Calculate the gradients of the loss with respect to the input data
    outputs = model(data)
    loss = nn.BCELoss()(outputs, labels)
    loss.backward()
    
    # Calculate the gradient of the loss with respect to the input data
    gradients = data.grad
    
    # Perform the modal evasion attack
    perturbed_data = data + epsilon * gradients.sign()
    
    return perturbed_data

# Define a function to train the model with or without the modal evasion attack
def train_model(model, data, labels, optimizer, criterion, num_epochs=50, attack=False):
    loss_values = []
    accuracy_values = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        if attack:
            perturbed_data = modal_evasion_attack(model, data, labels)
            outputs = model(perturbed_data).squeeze()
        else:
            outputs = model(data).squeeze()
        
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == labels.squeeze()).float().mean().item() * 100  # Calculate accuracy as percentage
        accuracy_values.append(int(accuracy))  # Store accuracy as integer percentage
        loss_values.append(loss.item())
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.0f}%")
    
    return loss_values, accuracy_values

# Initialize the model, criterion, and optimizer
model_with_attack = SimpleNN()
optimizer_with_attack = optim.Adam(model_with_attack.parameters(), lr=0.01)
model_without_attack = SimpleNN()
optimizer_without_attack = optim.Adam(model_without_attack.parameters(), lr=0.01)
criterion = nn.BCELoss()  # Define the criterion

# Train the model with and without the modal evasion attack
loss_values_with_attack, accuracy_values_with_attack = train_model(model_with_attack, data, labels, optimizer_with_attack, criterion, num_epochs=50, attack=True)
loss_values_without_attack, accuracy_values_without_attack = train_model(model_without_attack, data, labels, optimizer_without_attack, criterion, num_epochs=50, attack=False)

# Plot the accuracy and loss values
plt.figure(figsize=(12, 10))

# Plot accuracy
plt.subplot(2, 1, 1)
plt.plot(range(1, 51), accuracy_values_with_attack, label='With Modal Evasion Attack')
plt.plot(range(1, 51), accuracy_values_without_attack, label='Without Modal Evasion Attack')
plt.title('Training Accuracy with and without Modal Evasion Attack')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(2, 1, 2)
plt.plot(range(1, 51), loss_values_with_attack, label='With Modal Evasion Attack')
plt.plot(range(1, 51), loss_values_without_attack, label='Without Modal Evasion Attack')
plt.title('Training Loss with and without Modal Evasion Attack')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
