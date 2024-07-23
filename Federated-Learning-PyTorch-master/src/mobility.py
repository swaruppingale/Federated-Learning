#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

class User:
    def __init__(self, user_id, initial_location, availability_prob=0.9):
        self.user_id = user_id
        self.current_location = initial_location
        self.availability_prob = availability_prob

    def move(self):
        # Example: Randomly move user to a new location
        self.current_location = (np.random.uniform(-180, 180), np.random.uniform(-90, 90))
        print(self.current_location)
        

    def is_available(self):
        # Simulate availability based on location and availability probability
        return np.random.rand() < self.availability_prob

def simulate_user_availability(num_users, availability_prob=0.9):
    """Simulate user availability based on a given probability."""
    return np.random.rand(num_users) < availability_prob

if __name__ == '__main__':
    start_time = time.time()

    # Define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    # Load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Build model
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    global_model.to(device)
    global_model.train()
    print(global_model)

    # Copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # Initialize a log for user availability
    user_availability_log = []

    # Initialize users with random initial locations
    users = [User(user_id=i, initial_location=(np.random.uniform(-180, 180), np.random.uniform(-90, 90))) for i in range(args.num_users)]

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Simulate user availability
        user_availability = simulate_user_availability(args.num_users)
        user_availability_log.append(user_availability)

        for idx in idxs_users:
            if not user_availability[idx]:
                print(f'User {idx} is not available for this round due to connectivity issues.')
                continue

            # Move users to new locations (simulate mobility)
            users[idx].move()

            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        if not local_weights:
            print('No local weights available. Skipping this round.')
            continue

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    save_path = './save/objects/'
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.join(save_path, '{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))

    with open(file_name, 'wb') as f:
        pickle.dump({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'user_availability': user_availability_log}, f)
    print(f'Saved training results to {file_name}')
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    
    

    matplotlib.use('Agg')

    save_path_plots = '../save/'
    os.makedirs(save_path_plots, exist_ok=True)
    print(f"Save path for plots: {save_path_plots}")

    # Plot Training Loss vs Communication rounds
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    loss_plot_path = os.path.join(save_path_plots, 'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))
    plt.savefig(loss_plot_path)
    print(f'Saved training loss plot to {loss_plot_path}')
    plt.close()  # Close the figure to free memory

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    acc_plot_path = os.path.join(save_path_plots, 'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))
    plt.savefig(acc_plot_path)
    print(f'Saved accuracy plot to {acc_plot_path}')
    plt.close()  # Close the figure to free memory
