import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""
    
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)  # Convert idxs to a list for subscripting
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
class LocalUpdate:
    def __init__(self, args, dataset, idxs, logger=None, client_id=None, malicious_clients=None, manipulation_factor=None):
        self.args = args
        self.logger = logger
        self.train_loader, self.test_loader = self.train_test_loader(dataset, idxs)
        self.device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
        self.client_id = client_id
        self.malicious_clients = malicious_clients if malicious_clients else []
        self.manipulation_factor = manipulation_factor if manipulation_factor else 1.0

    def train_test_loader(self, dataset, idxs):
        """Returns train and test DataLoader objects for the local dataset"""
        train_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False)
        return train_loader, test_loader

    def update_weights(self, model, global_round):
        """Update the local model weights using local dataset"""
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.args.local_ep):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Simulate label flipping for malicious clients
                if self.client_id in self.malicious_clients:
                    target = self.flip_labels(target)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Evaluate local model after training
        model.eval()
        test_acc, test_loss = self.inference(model)
        return model.state_dict(), test_loss

    def flip_labels(self, target):
        """Flip a fraction of labels randomly"""
        num_samples = len(target)
        num_flips = int(self.manipulation_factor * num_samples)

        # Randomly choose indices to flip labels
        flip_indices = np.random.choice(num_samples, num_flips, replace=False)

        # Flip labels (assuming binary classification 0 and 1)
        flipped_labels = target.clone()
        flipped_labels[flip_indices] = 1 - flipped_labels[flip_indices]

        # Ensure labels are within the valid range
        flipped_labels = torch.clamp(flipped_labels, min=0, max=1)

        return flipped_labels

    def inference(self, model):
        """Evaluate the local model on test data"""
        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        return accuracy, test_loss

def test_inference(args, model, test_dataset):
    """Evaluate the global model on test dataset"""
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss
