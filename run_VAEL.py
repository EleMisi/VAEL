import argparse

import torch

from utils.addition_task_VAEL import run_vael

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run VAEL', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--device', action='store', nargs='?', const=None, default='cpu', type=str,
                        choices=None, help='Device. [default: cpu]', metavar=None)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    n_digits = 10

    # Experimental setting
    exp_setting = {
        'task': ['base'],
        'tag': [f'base_{n_digits}Digits'],
        'rec_loss': ['LAPLACE'],
        'max_epoch': [1],
        'n_exp': [2],
        'latent_dim_sub': [8],
        'latent_dim_sym': [15],
        'learning_rate': [1e-3],
        'dropout': [0.5],
        'dropout_ENC': [0.5],
        'dropout_DEC': [0.5],
        'recon_w': [1e-1],
        'kl_w': [1e-5],
        'query_w': [1.],
        'sup_w': [0.],
        'query': [True]}

    early_stopping_info = {
        'patience': 60,
        'delta': 1e-4}

    # Experiments and data folders
    exp_class = 'VAEL_' + exp_setting['rec_loss'][0]
    exp_folder = './experiments/'
    data_folder = "./data/MNIST/2mnist_{}digits/".format(n_digits)
    data_file = "2mnist_{}digits.pt".format(n_digits)
    # Load train and test images id for reproducibility
    train_label2idx = torch.load('./data/MNIST/2mnist_{}digits/train_label2idx.pt'.format(n_digits))
    test_label2idx = torch.load('./data/MNIST/2mnist_{}digits/test_label2idx.pt'.format(n_digits))

    # Run experiments
    run_vael(exp_setting, exp_class, exp_folder, data_folder, data_file, n_digits=n_digits, task=exp_setting['task'][0],
             tag=exp_setting['tag'][0], train_label2idx=train_label2idx, test_label2idx=test_label2idx, device=device,
             time_limit=100800, early_stopping_info=early_stopping_info, classes=None, time_delta=300)
