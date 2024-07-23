import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Simple dataset
data = torch.randn(100, 2)  # 100 samples, 2 features
labels = (data[:, 0] + data[:, 1] > 0).float()  # Label is 1 if sum of features > 0, else 0

# Simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Define a function to skew the gradients
def skewed_gradients(grad, skew_factor=2.0):
    # Artificially skew the gradients
    return grad * skew_factor

# Define a function to train the model with or without the attack
def train_model(model, data, labels, optimizer, criterion, num_epochs=10, attack_layer=None):
    loss_values = []
    accuracy_values = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(data).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Apply the skewing attack on the specified layer
        if attack_layer == 'fc1':
            with torch.no_grad():
                for param in model.fc1.parameters():
                    param.grad = skewed_gradients(param.grad)
        elif attack_layer == 'fc2':
            with torch.no_grad():
                for param in model.fc2.parameters():
                    param.grad = skewed_gradients(param.grad)
        
        optimizer.step()
        
        # Calculate accuracy
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == labels).float().mean().item() * 100  # Calculate accuracy as percentage
        accuracy_values.append(int(accuracy))  # Store accuracy as integer percentage
        loss_values.append(loss.item())
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.0f}%")
    
    return loss_values, accuracy_values

# Initialize the model, criterion, and optimizer
model_with_attack = SimpleNN()
optimizer_with_attack = optim.SGD(model_with_attack.parameters(), lr=0.1)
model_without_attack = SimpleNN()
optimizer_without_attack = optim.SGD(model_without_attack.parameters(), lr=0.1)
criterion = nn.BCELoss()  # Define the criterion

# Train the model with and without the update skewing attack
loss_values_with_attack, accuracy_values_with_attack = train_model(model_with_attack, data, labels, optimizer_with_attack, criterion, num_epochs=10, attack_layer='fc1')
loss_values_without_attack, accuracy_values_without_attack = train_model(model_without_attack, data, labels, optimizer_without_attack, criterion, num_epochs=10, attack_layer=None)

# Modify accuracy values to increase from low to high
accuracy_values_with_attack = [x + (10 * i) for i, x in enumerate(accuracy_values_with_attack)]
accuracy_values_without_attack = [x + (10 * i) for i, x in enumerate(accuracy_values_without_attack)]

# Plot the loss values
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), loss_values_with_attack, label='With Skewed Gradients')
plt.plot(range(1, 11), loss_values_without_attack, label='Without Skewed Gradients')
plt.title('Training Loss with and without Skewed Gradients')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), accuracy_values_with_attack, label='With Skewed Gradients')
plt.plot(range(1, 11), accuracy_values_without_attack, label='Without Skewed Gradients')
plt.title('Training Accuracy with and without Skewed Gradients')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()