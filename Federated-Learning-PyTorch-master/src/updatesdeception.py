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
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, client_id, malicious_clients=None, manipulation_factor=0.1):
        self.args = args
        self.dataset = dataset
        self.idxs = idxs
        self.logger = logger
        self.client_id = client_id
        self.malicious_clients = malicious_clients
        self.manipulation_factor = manipulation_factor
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        # Monitor decoy data handling
        self.decoy_handling = []

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
        decoy_count = 0

        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Track handling of decoy data
                if torch.equal(images, self.decoy_data) and labels.item() == self.decoy_labels:
                    self.decoy_handling.append((batch_idx, images, labels))
                    decoy_count += 1
                
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Return the inference loss and accuracy """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = nn.CrossEntropyLoss().to(self.device)
        for batch_idx, (images, labels) in enumerate(self.ldr_test):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

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
