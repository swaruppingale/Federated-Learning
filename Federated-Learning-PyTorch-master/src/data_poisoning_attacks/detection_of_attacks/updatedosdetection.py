import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset,Subset
import torch.optim as optim
import numpy as np


class LocalUpdate:
    def __init__(self, args, dataset, idxs, logger, client_id, malicious_clients, manipulation_factor):
        self.args = args
        self.dataset = dataset
        self.idxs = list(idxs)
        self.logger = logger
        self.client_id = client_id
        self.malicious_clients = malicious_clients
        self.manipulation_factor = manipulation_factor

        self.trainloader, self.validloader, self.testloader = self.train_val_test(self.dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation, and test data loaders.
        """
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)

        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                # Manipulate gradients if client is malicious
                if self.client_id in self.malicious_clients:
                    for param in model.parameters():
                        param.grad *= self.manipulation_factor

                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = nn.CrossEntropyLoss().to(self.device)
        for batch_idx, (images, labels) in enumerate(self.testloader):
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

    def calculate_distance(self, global_model, local_weights):
        distance = 0
        for key in global_model.state_dict().keys():
            distance += torch.norm(global_model.state_dict()[key] - local_weights[key]) ** 2
        return torch.sqrt(distance).item()

    def detect_anomaly(self, distance, threshold):
        return distance > threshold
    

def test_inference(args, model, test_dataset):
        
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

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
