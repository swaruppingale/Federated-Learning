import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset,Subset
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
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, client_id, malicious_clients, manipulation_factor):
        self.args = args
        self.logger = logger
        self.device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
        self.dataset = dataset
        self.idxs = list(idxs)
        self.client_id = client_id
        self.malicious_clients = malicious_clients
        self.manipulation_factor = manipulation_factor

        # Initialize the data loader
        self.train_loader = DataLoader(Subset(dataset, self.idxs), batch_size=self.args.local_bs, shuffle=True)
        self.criterion = nn.NLLLoss().to(self.device)

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # Manipulate gradients if the client is malicious
        if self.client_id in self.malicious_clients:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad *= self.manipulation_factor

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0, 0
        criterion = nn.NLLLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                batch_loss = criterion(outputs, labels)
                loss += batch_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy, loss

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
