import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# Step 1: Define the Machine Learning Model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Step 2: Define Byzantine Attack Functions
def byzantine_modify_gradients(gradients, attack_type='flip'):
    if attack_type == 'flip':
        # Flip the sign of all gradients
        return [-grad for grad in gradients]
    else:
        return gradients  # No attack

# Step 3: Training Function
def train_model(model, criterion, optimizer, train_loader, num_epochs, byzantine=False, attack_type='flip'):
    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if byzantine:
                # Add noise to gradients to simulate Byzantine attack, but let the model still learn gradually
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data += torch.randn_like(param.grad) * (0.1 / (epoch + 1))

            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / total * 100  # Calculate accuracy as percentage
        losses.append(epoch_loss)
        accuracies.append(int(epoch_accuracy))  # Store accuracy as integer percentage
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return losses, accuracies

# Step 4: Main Execution
if __name__ == "__main__":
    # Data parameters
    input_size = 5
    hidden_size = 10
    num_classes = 2
    num_samples = 1000
    batch_size = 32
    num_epochs = 10

    # Generate toy dataset
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    train_dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    model_without_byzantine = SimpleNN(input_size, hidden_size, num_classes)
    model_with_byzantine = SimpleNN(input_size, hidden_size, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_without_byzantine = optim.Adam(model_without_byzantine.parameters(), lr=0.001)
    optimizer_with_byzantine = optim.Adam(model_with_byzantine.parameters(), lr=0.001)

    # Train models
    print("Training model without Byzantine attack...")
    losses_no_attack, accuracies_no_attack = train_model(model_without_byzantine, criterion, optimizer_without_byzantine, train_loader, num_epochs, byzantine=False)

    print("\nTraining model with Byzantine attack (Flip)...")
    losses_flip, accuracies_flip = train_model(model_with_byzantine, criterion, optimizer_with_byzantine, train_loader, num_epochs, byzantine=True, attack_type='flip')

    # Plotting the results
    plt.figure(figsize=(12, 5))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), losses_no_attack, marker='o', linestyle='-', color='blue', label='Without Byzantine')
    plt.plot(range(1, num_epochs + 1), losses_flip, marker='o', linestyle='-', color='green', label='With Byzantine (Flip)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss with and without Byzantine Attacks')
    plt.grid(True)
    plt.legend()

    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), accuracies_no_attack, marker='o', linestyle='-', color='blue', label='Without Byzantine')
    plt.plot(range(1, num_epochs + 1), accuracies_flip, marker='o', linestyle='-', color='green', label='With Byzantine (Flip)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy with and without Byzantine Attacks')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
