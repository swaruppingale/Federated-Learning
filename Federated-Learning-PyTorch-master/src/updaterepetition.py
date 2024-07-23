# updates.py

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset,device, idxs, logger, client_id=None, malicious_clients=None, manipulation_factor=0.0):
        self.args = args
        self.device = device
        self.dataset = dataset
        self.idxs = list(idxs)
        self.logger = logger
        self.client_id = client_id
        self.manipulation_factor = manipulation_factor
        self.malicious_clients = malicious_clients if malicious_clients is not None else []
        self.ldr_train = self.train_dataloader()

    def train_dataloader(self):
        return DataLoader(DatasetSplit(self.dataset, self.idxs), batch_size=self.args.local_bs, shuffle=True)

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # Apply data repetition attack
                if self.client_id in self.malicious_clients:
                    images, labels = self.data_repetition_attack(images, labels)

                model.zero_grad()
                log_probs = model(images)
                loss = torch.nn.functional.cross_entropy(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        model.eval()
        loss = 0
        total = 0
        correct = 0

        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            outputs = model(images)
            batch_loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss += batch_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy, loss

    def data_repetition_attack(self, images, labels):
        # Apply data repetition to a subset of the images
        n_repeats = int(len(images) * self.manipulation_factor)
        repeat_idxs = np.random.choice(len(images), n_repeats, replace=True)
        repeated_images = images[repeat_idxs]
        repeated_labels = labels[repeat_idxs]

        images = torch.cat((images, repeated_images), 0)
        labels = torch.cat((labels, repeated_labels), 0)
        return images, labels
