#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


# class LocalUpdate(object):
#     def __init__(self, args, dataset, idxs, logger):
#         self.args = args
#         self.logger = logger
#         self.trainloader, self.validloader, self.testloader = self.train_val_test(
#             dataset, list(idxs))
#         #self.device = 'cuda' if args.gpu else 'cpu'
#         self.device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
#         # Default criterion set to NLL loss function
#         self.criterion = nn.NLLLoss().to(self.device)

class LocalUpdate:
    def __init__(self, args, dataset, idxs, logger, client_id=None, malicious_clients=None, manipulation_factor=None):
        self.args = args
        self.dataset = dataset
        self.idxs = list(idxs)
        self.logger = logger
        self.client_id = client_id
        self.malicious_clients = malicious_clients if malicious_clients is not None else []
        self.manipulation_factor = manipulation_factor if manipulation_factor is not None else 1.0
        self.device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.data_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, self.idxs), batch_size=self.args.local_bs, shuffle=True)

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                
                # Manipulate gradients for malicious clients
                if self.client_id in self.malicious_clients:
                    for param in model.parameters():
                        param.grad *= self.manipulation_factor
                
                optimizer.step()
                
                if self.args.verbose and batch_idx % 10 == 0:
                    print(f'Update Epoch: {epoch} [{batch_idx * len(images)}/{len(self.data_loader.dataset)}'
                          f' ({100. * batch_idx / len(self.data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        model.eval()
        total, correct = 0, 0
        loss = 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.data_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy, loss / len(self.data_loader)


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    # def update_weights(self, model, global_round):
    #     # Set mode to train model
    #     model.train()
    #     epoch_loss = []

    #     # Set optimizer for the local updates
    #     if self.args.optimizer == 'sgd':
    #         optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
    #                                     momentum=0.5)
    #     elif self.args.optimizer == 'adam':
    #         optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
    #                                      weight_decay=1e-4)

    #     for iter in range(self.args.local_ep):
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.trainloader):
    #             images, labels = images.to(self.device), labels.to(self.device)

    #             model.zero_grad()
    #             log_probs = model(images)
    #             loss = self.criterion(log_probs, labels)
    #             loss.backward()
    #             optimizer.step()

    #             if self.args.verbose and (batch_idx % 10 == 0):
    #                 print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                     global_round, iter, batch_idx * len(images),
    #                     len(self.trainloader.dataset),
    #                     100. * batch_idx / len(self.trainloader), loss.item()))
    #             self.logger.add_scalar('loss', loss.item())
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))

    #     return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # def inference(self, model):
    #     """ Returns the inference accuracy and loss.
    #     """

    #     model.eval()
    #     loss, total, correct = 0.0, 0.0, 0.0

    #     for batch_idx, (images, labels) in enumerate(self.testloader):
    #         images, labels = images.to(self.device), labels.to(self.device)

    #         # Inference
    #         outputs = model(images)
    #         batch_loss = self.criterion(outputs, labels)
    #         loss += batch_loss.item()

    #         # Prediction
    #         _, pred_labels = torch.max(outputs, 1)
    #         pred_labels = pred_labels.view(-1)
    #         correct += torch.sum(torch.eq(pred_labels, labels)).item()
    #         total += len(labels)

    #     accuracy = correct/total
    #     return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    #device = 'cuda' if args.gpu else 'cpu'
    criterion = torch.nn.CrossEntropyLoss().to(device)
    #criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

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

    accuracy = correct/total
    return accuracy, loss
