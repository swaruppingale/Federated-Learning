import os
import copy
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

from attachments import args_parser
from updatess import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


def detect_sybil_attack(local_weights, threshold=0.05):
    """
    Detects potential Sybil attacks based on the similarity of local model updates.

    Args:
    - local_weights (list): List of local model weights from different clients.
    - threshold (float): Threshold for detecting similarity in updates.

    Returns:
    - List of indices of potential Sybil clients.
    """
    def cosine_similarity(w1, w2):
        dot_product = np.dot(w1, w2)
        norm_w1 = np.linalg.norm(w1)
        norm_w2 = np.linalg.norm(w2)
        return dot_product / (norm_w1 * norm_w2)

    malicious_clients = set()

    weight_vectors = [np.concatenate([w.flatten() for w in client_weights.values()]) for client_weights in local_weights]
    num_clients = len(weight_vectors)

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            sim = cosine_similarity(weight_vectors[i], weight_vectors[j])
            if sim > threshold:
                malicious_clients.add(i)
                malicious_clients.add(j)

    return list(malicious_clients)


def train_federated(args, train_dataset, test_dataset, user_groups, device, logger, malicious_clients=None):
    # Build global model
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

    # Training parameters
    train_loss, train_accuracy = [], []
    print_every = 2

    # Federated Learning
    for epoch in tqdm(range(args.epochs), desc="", unit="epoch"):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            # Initialize LocalUpdate for each user
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger,
                                      client_id=idx, malicious_clients=malicious_clients, manipulation_factor=0.1)
            # Update local model weights
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # Detecting Sybil attacks
        sybil_clients = detect_sybil_attack(local_weights, threshold=0.95)
        if sybil_clients:
            print(f"Potential Sybil attack detected in round {epoch + 1}: Clients {sybil_clients}")

            # Further action (optional): Remove or isolate sybil_clients from further training rounds

        # Average local model weights to update global model
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # Calculates and stores the average training loss for the current round.
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Evaluate global model on training data by aggregating accuracy and loss from all users.
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

    return train_loss, train_accuracy, global_model


def main():
    start_time = time.time()

    # Define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs')  # Adjust the path as needed

    # Parse command-line arguments
    args = args_parser()
    exp_details(args)

    # Set device (GPU if available, otherwise CPU)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    # Load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Training without Sybil attacks
    print("Training without Sybil attacks...")
    train_loss_no_sybil, train_accuracy_no_sybil, _ = train_federated(
        args, train_dataset, test_dataset, user_groups, device, logger, malicious_clients=[]
    )

    # Training with Sybil attacks (Example: Assuming clients 0, 1 are Sybils)
    print("Training with Sybil attacks...")
    train_loss_sybil, train_accuracy_sybil, _ = train_federated(
        args, train_dataset, test_dataset, user_groups, device, logger, malicious_clients=[0, 1]
    )

    # Plot Training Loss vs Communication rounds with and without Sybil detection
    plt.figure()
    plt.title('Training Loss vs Communication Rounds')
    plt.plot(range(len(train_loss_no_sybil)), train_loss_no_sybil, color='r', label='Without Sybil Detection')
    plt.plot(range(len(train_loss_sybil)), train_loss_sybil, color='b', label='With Sybil Detection')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.legend()
    loss_plot_path = './save/plots/sybil_training_loss_comparison.png'
    os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
    plt.savefig(loss_plot_path)
    print(f'Saved training loss comparison plot to {loss_plot_path}')
    plt.close()

    # Plot Average Accuracy vs Communication rounds with and without Sybil detection
    plt.figure()
    plt.title('Average Accuracy vs Communication Rounds')
    plt.plot(range(len(train_accuracy_no_sybil)), train_accuracy_no_sybil, color='r', label='Without Sybil Detection')
    plt.plot(range(len(train_accuracy_sybil)), train_accuracy_sybil, color='b', label='With Sybil Detection')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.legend()
    acc_plot_path = './save/plots/sybil_training_accuracy_comparison.png'
    plt.savefig(acc_plot_path)
    print(f'Saved accuracy comparison plot to {acc_plot_path}')
    plt.close()

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
