import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Define the SimpleCNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the transform for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the training and test datasets
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

# Define a function to add a backdoor trigger to the data
def add_backdoor(data, target, trigger_value=1.0):
    if data.dim() == 2:
        data = data.unsqueeze(0)
    
    data[:, 26:28, 26:28] = trigger_value
    target = 0
    return data, target

# Apply the backdoor to a subset of the training dataset
poisoned_indices = np.random.choice(len(train_dataset), int(0.1 * len(train_dataset)), replace=False)
backdoor_train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
for idx in poisoned_indices:
    backdoor_train_dataset.data[idx], backdoor_train_dataset.targets[idx] = add_backdoor(backdoor_train_dataset.data[idx].float()/255, backdoor_train_dataset.targets[idx])

# Create DataLoaders for the training and test datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
backdoor_train_loader = torch.utils.data.DataLoader(backdoor_train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize the model, optimizer, and loss function
def initialize_model():
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer

criterion = nn.CrossEntropyLoss()

# Track metrics
train_losses = {'clean': [], 'backdoor': []}
test_accuracies = {'clean': [], 'backdoor': []}

# Define the training function
def train(model, device, train_loader, optimizer, epoch, key):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    train_losses[key].append(avg_loss)
    print(f'Train Epoch: {epoch} \tLoss ({key}): {avg_loss:.6f}')

# Define the testing function
def test(model, device, test_loader, key):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total * 100
    test_accuracies[key].append(int(accuracy))  # Store accuracy as integer percentage
    print(f'Test Accuracy ({key}): {accuracy:.2f}%')

# Train and test the model with clean data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, optimizer = initialize_model()
model.to(device)

for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, epoch, 'clean')
    test(model, device, test_loader, 'clean')

# Train and test the model with backdoor data
model, optimizer = initialize_model()
model.to(device)

for epoch in range(1, 6):
    train(model, device, backdoor_train_loader, optimizer, epoch, 'backdoor')
    test(model, device, test_loader, 'backdoor')

# Prepare backdoor test data and DataLoader
backdoor_test_data = test_dataset.data.float()/255
backdoor_test_target = test_dataset.targets.clone()

for idx in range(len(backdoor_test_data)):
    backdoor_test_data[idx], backdoor_test_target[idx] = add_backdoor(backdoor_test_data[idx], backdoor_test_target[idx])

backdoor_test_dataset = torch.utils.data.TensorDataset(backdoor_test_data.unsqueeze(1), backdoor_test_target)
backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_dataset, batch_size=1000, shuffle=False)

# Test the model on backdoor test data
test(model, device, backdoor_test_loader, 'backdoor')

# Define epochs variable for plotting
epochs = range(1, 6)

# Ensure same number of epochs for plotting
if len(test_accuracies['backdoor']) > len(epochs):
    test_accuracies['backdoor'] = test_accuracies['backdoor'][:len(epochs)]

# Plot the training loss for clean and backdoor data
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epochs, train_losses['clean'], marker='o', color=color, label='Clean Training Loss')
ax1.plot(epochs, train_losses['backdoor'], marker='x', color='tab:orange', label='Backdoor Training Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

fig.tight_layout()
plt.title('Training Loss (Clean vs. Backdoor)')
plt.show()

# Plot the test accuracy for clean and backdoor data
fig, ax2 = plt.subplots(figsize=(12, 6))
 
color = 'tab:green'
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(epochs, test_accuracies['clean'], marker='o', color=color, label='Clean Test Accuracy')
ax2.plot(epochs, test_accuracies['backdoor'], marker='x', color='tab:red', label='Backdoor Test Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper left')

fig.tight_layout()
plt.title('Test Accuracy (Clean vs. Backdoor)')
plt.show()

print("Backdoor test loader successfully created.")
print("Backdoor applied to training data and test loader created successfully.")
