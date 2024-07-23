import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib
import pickle

from attachments import args_parser
from updatedosdetection import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

print("Training withous DoS attack Detection")

# Function to run federated learning and return the training accuracy and loss
def federated_learning(args, device, train_dataset, test_dataset, user_groups, logger, malicious_clients, manipulation_factor, anomaly_threshold=0.5):
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

    # Copy weights
    global_weights = global_model.state_dict()

    # Training
    train_accuracy = []
    train_loss = []
    anomalies_detected = []
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_distances = [], [], []

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = local_updates[idx]
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            
            distance = local_model.calculate_distance(global_model, w)
            local_distances.append(distance)
            if local_model.detect_anomaly(distance, anomaly_threshold):
                anomalies_detected.append((epoch, idx))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # Calculate avg training accuracy and loss over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = local_updates[c]
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))
        train_loss.append(sum(list_loss) / len(list_loss))

    return train_accuracy, train_loss, anomalies_detected

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

    # Accuracy and loss without DoS attack
    accuracy_no_attack, loss_no_attack, anomalies_no_attack = federated_learning(
        args, device, train_dataset, test_dataset, user_groups, logger, malicious_clients=[], manipulation_factor=1.0)

    # Accuracy and loss with DoS attack
    malicious_clients = [0, 1]  # Specify indices of malicious clients
    manipulation_factor = 10.0  # Factor by which gradients will be scaled for malicious clients
    accuracy_with_attack, loss_with_attack, anomalies_with_attack = federated_learning(
        args, device, train_dataset, test_dataset, user_groups, logger, malicious_clients=malicious_clients, manipulation_factor=manipulation_factor)

    # Plotting
    matplotlib.use('Agg')

    # Plot Average Accuracy vs Communication rounds for both scenarios
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(accuracy_no_attack)), accuracy_no_attack, color='g', label='No DoS Attack Detection')
    plt.plot(range(len(accuracy_with_attack)), accuracy_with_attack, color='r', label='With DoS Attack Detection')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.legend()

    acc_comparison_plot_path = './save/plots/DoS_attack_vs_no_attack_accuracy.png'
    os.makedirs(os.path.dirname(acc_comparison_plot_path), exist_ok=True)
    plt.savefig(acc_comparison_plot_path)
    print(f'Saved accuracy comparison plot to {acc_comparison_plot_path}')
    plt.close()

    # Plot Training Loss vs Communication rounds for both scenarios
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(loss_no_attack)), loss_no_attack, color='g', label='No DoS Attack Detection')
    plt.plot(range(len(loss_with_attack)), loss_with_attack, color='r', label='With DoS Attack Detection')
    plt.ylabel('Training Loss')
    plt.xlabel('Communication Rounds')
    plt.legend()

    loss_comparison_plot_path = './save/plots/DoS_attack_vs_no_attack_loss.png'
    os.makedirs(os.path.dirname(loss_comparison_plot_path), exist_ok=True)
    plt.savefig(loss_comparison_plot_path)
    print(f'Saved loss comparison plot to {loss_comparison_plot_path}')
    plt.close()

    # Print final results
    print(f'\n Results without DoS attack after {args.epochs} global rounds of training:')
    print("|---- Final Train Accuracy: {:.2f}%".format(100 * accuracy_no_attack[-1]))
    print("|---- Final Train Loss: {:.4f}".format(loss_no_attack[-1]))

    print(f'\n Results with DoS attack after {args.epochs} global rounds of training:')
    print("|---- Final Train Accuracy: {:.2f}%".format(100 * accuracy_with_attack[-1]))
    print("|---- Final Train Loss: {:.4f}".format(loss_with_attack[-1]))

    # print(f'\nAnomalies detected without DoS attack: {anomalies_no_attack}')
    print(f'Anomalies detected with DoS attack: {anomalies_with_attack}')

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
