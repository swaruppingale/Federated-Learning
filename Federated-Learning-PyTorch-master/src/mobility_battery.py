# try3.py

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

    

    # Build model based on arguments
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
    # Define battery mobility (example)
    battery_mobility = {0: True, 1: True, 2: False, 3: True, 4: False} 

    # Copy initial weights
    global_weights = global_model.state_dict()

    # Training and evaluation
    train_loss, train_accuracy = [], []
    print_every = 2

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger,
                                      client_id=idx, battery_mobility=battery_mobility)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate average training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[c], logger=logger,
                                      client_id=c, battery_mobility=battery_mobility)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Evaluate on test dataset
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Save training results
    save_path = './save/objects/'
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.join(save_path, '{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))

    with open(file_name, 'wb') as f:
        pickle.dump({'train_loss': train_loss, 'train_accuracy': train_accuracy}, f)
    print(f'Saved training results to {file_name}')
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # Plot Training Loss vs Communication rounds
    matplotlib.use('Agg')  # Ensure matplotlib backend is set to Agg for non-interactive plotting
    save_path_plots = '../save/'
    os.makedirs(save_path_plots, exist_ok=True)

    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.title('Training Loss vs Communication rounds')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Training Loss')
    loss_plot_path = os.path.join(save_path_plots, 'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))
    plt.savefig(loss_plot_path)
    plt.close()

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.title('Average Accuracy vs Communication rounds')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Average Accuracy')
    acc_plot_path = os.path.join(save_path_plots, 'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))
    plt.savefig(acc_plot_path)
    plt.close()
