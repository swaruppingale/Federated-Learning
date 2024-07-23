#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of rounds of training (default: 10)')
    # parser.add_argument('--epochs', type=int, default=10, help="Number of training rounds")
    parser.add_argument('--num_users', type=int, default=100, help="Number of users")
    parser.add_argument('--frac', type=float, default=0.1, help='Fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10, help="Number of local epochs")
    parser.add_argument('--local_bs', type=int, default=10, help="Local batch size")
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
    parser.add_argument('--model', type=str, default='cnn', help='Model type (mlp, cnn)')
    parser.add_argument('--dataset', type=str, default='mnist', help="Dataset (mnist, fmnist, cifar)")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use")
    parser.add_argument('--verbose', action='store_true', help="Print verbose output")
    parser.add_argument('--manipulation_factor', type=float, default=0.1, help="factor by which to manipulate data for attacks")
    
    
    
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    # parser.add_argument('--dataset', type=str, default='mnist', help="name \
    #                     of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    # parser.add_argument('--gpu',  default=None, help="To use cuda, set \
    #                     to a specific GPU ID. Default set to use CPU.")
    
    # parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use")

    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    # parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
