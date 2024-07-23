import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib,pickle

from attach import args_parser
from updatesinglepixel import LocalUpdate,test_inference  
from models_one import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utilities import get_dataset, average_weights, exp_details


# main.py

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

    # Define malicious clients and gradient manipulation settings
    malicious_clients = [0, 1]  # Specify indices of malicious clients
    manipulation_factor = 10.0  # Factor by which gradients will be scaled for malicious clients

    # Initialize user_groups with LocalUpdate instances
    local_updates = {}
    for client_id in range(args.num_users):
        local_updates[client_id] = LocalUpdate(
            args=args,
            dataset=train_dataset,
            idxs=user_groups[client_id],
            logger=logger,
            client_id=client_id,
            malicious_clients=malicious_clients,
            manipulation_factor=manipulation_factor
        )

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

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = local_updates[idx]
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = local_updates[c]
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
    

    matplotlib.use('Agg')

    

    # Plot Training Loss vs Communication rounds
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    
    loss_plot_path = './save/plots/DoS_attack_loss.png'
    os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
    plt.savefig(loss_plot_path)
    print(f'Saved training loss comparison plot to {loss_plot_path}')
    plt.close()
    

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    
    acc_plot_path = './save/plots/DoS_training_accuracy.png'
    plt.savefig(acc_plot_path)
    print(f'Saved accuracy comparison plot to {acc_plot_path}')
    plt.close()

print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))