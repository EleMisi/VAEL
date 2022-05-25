import os.path
import random
from datetime import datetime
from itertools import product
from math import isnan
from pathlib import Path
from time import time, sleep

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from torch import nn, optim

import const_define as cd
from utils.mario_utils.mario_dataset import MarioDataset
from utils.mario_utils.metrics import reconstructive_ability, discriminative_ability, generative_ability
from utils.mario_utils.plot_utils import learning_curve, conditional_image_generation, image_reconstruction, \
    image_generation
from utils.mario_utils.train import train
from models.vael import MarioVAELModel
from models.vael_networks import MarioEncoder, MarioDecoder, MarioMLP

# TODO: Documentation

def lock_resource(lock_filename):
    with open(lock_filename, 'w') as f:
        f.write('locked')


def release_lock(lock_filename):
    os.remove(lock_filename)


def update_resource(log_filepath, update_info, lock_filename='access.lock'):
    # {'Experiment_ID': 0, 'Run_ID': 1, ...}
    print('Updating resource with: {}'.format(update_info))

    # Check if lock file does exist
    # If it exists -> I have to wait (sleep -> 1.0 second)
    while os.path.isfile(lock_filename):
        sleep(1.0)

    # Do lock
    lock_resource(lock_filename)

    # Do update
    try:
        log_file = open(log_filepath, 'a')
        log_file.write(update_info)
        log_file.close()
    except Exception as e:
        raise e
    finally:
        # Release lock
        release_lock(lock_filename)


def load_data(batch_size, data_folder, seeds, task='base', tag='base'):
    datasets=dict()
    if task == 'base':
        print("\n Base task\n")
        datasets = {}
        for mode in ['train', 'val', 'test']:
            datasets[mode] = MarioDataset(path=data_folder, batch_size=batch_size[mode], seed=seeds[mode],
                                          mode=mode)
            print("{} set: batches of {} images ({} images)".format(mode, batch_size[mode],
                                                                    len(datasets[mode])))

    return datasets


def load_classifier(checkpoint_path, device, hidden_sizes=32, output_size=9):
    # Build a feed-forward network
    hidden_sizes = 32
    output_size = 9

    # 5,2,3 -> 34
    # 5,2,3 -> 12
    # 5,1,3 -> 4
    clf = nn.Sequential(nn.Conv2d(in_channels=3,
                                  out_channels=hidden_sizes,
                                  kernel_size=5,
                                  stride=3,
                                  padding=2),
                        nn.SELU(),
                        nn.Dropout(p=0.5),
                        nn.Conv2d(in_channels=hidden_sizes,
                                  out_channels=hidden_sizes * 2,
                                  kernel_size=5,
                                  stride=3,
                                  padding=2),
                        nn.SELU(),
                        nn.Dropout(p=0.5),
                        nn.Conv2d(in_channels=hidden_sizes * 2,
                                  out_channels=hidden_sizes * 4,
                                  kernel_size=5,
                                  stride=3,
                                  padding=1),
                        nn.SELU(),
                        nn.Dropout(p=0.5),
                        nn.Flatten(start_dim=1, end_dim=-1),
                        nn.Linear(hidden_sizes * 4 * 4 * 4, output_size))

    clf = clf.to(device)

    if torch.cuda.is_available():
        clf.load_state_dict(torch.load(checkpoint_path)['model'])
        clf = clf.to(device)
    else:
        clf.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['model'])

    return clf


def define_experiment(exp_folder, exp_class, params, exp_counter):
    log_file = Path(os.path.join(exp_folder, exp_class, exp_class + '.csv'))
    params_columns = ['latent_dim_sub',
                      'latent_dim_sym',
                      'learning_rate',
                      'dropout',
                      'dropout_ENC',
                      'dropout_DEC',
                      'recon_w',
                      'kl_w',
                      'query_w',
                      'sup_w',
                      'max_epoch',
                      'hidden_chs_MLP',
                      'hidden_chs_ENC',
                      'hidden_chs_DEC']
    if log_file.is_file():
        # Load file
        log_csv = pd.read_csv(os.path.join(exp_folder, exp_class, exp_class + '.csv'))

        # Check if the required number of test has been already satisfied
        required_exp = params['n_exp']

        if len(log_csv) > 0:
            query = ''.join(f' {key} == {params[key]} &' for key in params_columns)[:-1]
            n_exp = len(log_csv.query(query))
            if n_exp == 0:
                exp_ID = log_csv['exp_ID'].max() + 1
                if isnan(exp_ID):
                    exp_ID = 1
                counter = required_exp - n_exp
                print("\n\n{} compatible experiments found in file {} -> {} experiments to run.".format(n_exp,
                                                                                                        os.path.join(
                                                                                                            exp_folder,
                                                                                                            exp_class,
                                                                                                            exp_class + '.csv'),
                                                                                                        counter))

                run_ID = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
            elif n_exp < required_exp:
                exp_ID = log_csv.query(query)['exp_ID'].values[0]
                counter = required_exp - n_exp
                print("\n\n{} compatible experiments found in file {} -> {} experiments to run.".format(n_exp,
                                                                                                        os.path.join(
                                                                                                            exp_folder,
                                                                                                            exp_class,
                                                                                                            exp_class + '.csv'),
                                                                                                        counter))

                run_ID = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')

            else:
                print("\n\n{} compatible experiments found in file {} -> No experiments to run.".format(n_exp,
                                                                                                        os.path.join(
                                                                                                            exp_folder,
                                                                                                            exp_class,
                                                                                                            exp_class + '.csv'),
                                                                                                        0))
                counter = 0
                exp_ID = log_csv.query(query)['exp_ID'].values[0]
                run_ID = None
        else:
            counter = required_exp
            exp_ID = log_csv['exp_ID'].max() + 1
            if isnan(exp_ID):
                exp_ID = 1
            run_ID = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
            print("\n\n0 compatible experiments found in file {} -> {} experiments to run.".format(
                exp_folder + exp_class + '.csv',
                counter))





    else:
        counter = params['n_exp']
        # Create log file
        log_file = open(os.path.join(exp_folder, exp_class, exp_class + '.csv'), 'w')
        header = 'exp_ID,run_ID,' + ''.join(str(key) + ',' for key in params_columns) + params[
            'rec_loss'] + "_recon_val,f1_val," + params[
                     'rec_loss'] + "_recon_train,f1_train,acc_gen,train_elbo,val_elbo,epochs,max_epoch,time,tag\n"
        log_file.write(header)
        # Define experiment ID
        exp_ID = 1
        run_ID = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
        print()
        print('-' * 40)
        print("\nNO csv file found -> new file created {}".format(
            os.path.join(exp_folder, exp_class, exp_class + '.csv')))
        print('-' * 40)
        print()
        log_file.close()

    exp_counter += 1
    print()
    print('*' * 40)
    print("Running exp {} (exp ID: {})".format(exp_counter, exp_ID))
    print("Parameters:", params)
    print('*' * 40)
    print()

    return run_ID, str(exp_ID), exp_counter, counter, params_columns


def build_worlds_queries_matrix(sequence_len, n_digits):
    """Build Worlds-Queries matrix"""
    possible_worlds = list(product(range(n_digits), repeat=sequence_len))
    n_worlds = len(possible_worlds)
    n_queries = len(range(0, n_digits + n_digits))
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
    w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
    for w in range(n_worlds):
        digit1, digit2 = look_up[w]
        for q in range(n_queries):
            if digit1 + digit2 == q:
                w_q[w, q] = 1
    return w_q


def run_mario_vael(param_grid, exp_class, exp_folder, data_folder, batch_size, task='base', tag='base', device='cpu',
                   time_limit=500, early_stopping_info=None, time_delta=350):
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print("\nDevice:", device)

    # Load data
    # No validation or test set,  only training set
    seeds = {}
    seeds['train'] = 88888
    seeds['val'] = 99999
    seeds['test'] = 77777
    # Check whether dataset exists, if not build it
    datasets = load_data(batch_size=batch_size, data_folder=data_folder,
                         seeds=seeds, task=task, tag=tag)

    # Define pre-compiled ProbLog programs and worlds-queries matrix
    folder_path = os.path.join(cd.PROJECT_DIR,'utils', 'mario_utils', 'problog_matrices', f'3x3')
    W_all = torch.tensor(torch.load(os.path.join(folder_path, "W_all.pt"))).type(torch.float32).to(device)
    WQ_all = torch.load(os.path.join(folder_path, "WQ_all.pt")).type(torch.float32).to(device)
    W_adm = torch.tensor(torch.load(os.path.join(folder_path, "W_adm.pt"))).type(torch.float32).to(device)
    WQ_adm = torch.load(os.path.join(folder_path, "WQ_adm.pt")).type(torch.float32).to(device)

    # Load Mario classifier for generative ability evaluation
    clf = load_classifier(
        checkpoint_path=os.path.join(cd.PROJECT_DIR,'utils', 'mario_utils', 'mario_classifier', 'mario_position_clf.pt'), device=device)

    # Start experiments
    start_exp = time()
    elapsed = 0
    tot_number_exp = 0
    param_list = list(ParameterGrid(param_grid))

    while elapsed + time_delta < time_limit:

        Path(os.path.join(exp_folder, exp_class)).mkdir(parents=True, exist_ok=True)
        exp_counter = 0
        # Sample one possible parameters config and remove it from the experiments list
        try:
            config = random.sample(param_list, k=1)[0]
        except:
            print("\n\nNo more experiments to run.")
            break
        param_list.remove(config)

        counter = 1
        while counter and elapsed + time_delta < time_limit:

            # Check if the required number of test for the current configuration has been already satisfied
            run_ID, exp_ID, exp_counter, counter, params_columns = define_experiment(exp_folder, exp_class, config,
                                                                                     exp_counter)
            if not counter:
                break

            # Build VAEL model
            encoder = MarioEncoder(img_channels=3, hidden_channels=config['hidden_chs_ENC'],
                                   latent_dim=config['latent_dim_sym'] + config['latent_dim_sub'], dropout=config['dropout_ENC'])
            decoder = MarioDecoder(img_channels=3, hidden_channels=config['hidden_chs_DEC'],
                                   latent_dim_sub=config['latent_dim_sub'],
                                   label_dim=9, dropout=config['dropout_DEC'])
            mlp = MarioMLP(in_features=config['latent_dim_sym'], n_facts=18, hidden_channels=config['hidden_chs_MLP'])
            model = MarioVAELModel(encoder, decoder, mlp,
                                   latent_dims=(config['latent_dim_sym'], config['latent_dim_sub']), model_dict=None,
                                   WQ_all=WQ_all, W_all=W_all, WQ_adm=WQ_adm, W_adm=W_adm, grid_dims=(3, 3),
                                   dropout=config['dropout'], is_train=True, device=device)

            model = model.to(device)

            optimizer_lagrangian = None
            recon_w = config['recon_w']
            query_w = config['query_w']
            kl_w = config['kl_w']

            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

            # Timing
            start = time()

            # Train
            checkpoint_path, epoch, train_info, validation_info = train(model,
                                                                        optimizer,
                                                                        n_epochs=config['max_epoch'],
                                                                        train_set=datasets['train'],
                                                                        val_set=datasets['val'],
                                                                        early_stopping_info=early_stopping_info,
                                                                        run_ID=str(run_ID),
                                                                        recon_w=recon_w,
                                                                        kl_w=kl_w,
                                                                        query_w=query_w,
                                                                        sup_w=config['sup_w'],
                                                                        folder=os.path.join(exp_folder, exp_class,
                                                                                            exp_ID),
                                                                        rec_loss=config['rec_loss'])
                                                                        
                                                                        
            train_elbo = train_info['true_elbo'][epoch-1]
            val_elbo = validation_info['true_elbo'][epoch-1]                                         

            # Timing
            end = time()
            tot_time = end - start

            # Save training and validation info
            np.save(os.path.join(exp_folder, exp_class, exp_ID, str(run_ID), 'train_info.npy'), train_info)
            np.save(os.path.join(exp_folder, exp_class, exp_ID, str(run_ID), 'validation_info.npy'), validation_info)

            # Load checkpoint
            last_checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(last_checkpoint['model'])
            optimizer.load_state_dict(last_checkpoint['optimizer'])

            # Evaluation
            folder = os.path.join(exp_folder, exp_class, exp_ID)
            model.eval()
            with torch.no_grad():

                print("\n\n****TEST MODEL WITH CONFIGURATION*****\n", config)

                print("   Evaluating reconstructive ability...")
                recon_acc_val_set = reconstructive_ability(model, datasets['val'], config['rec_loss'])
                # #view_used_mem()

                recon_acc_train_set = reconstructive_ability(model, datasets['train'], config['rec_loss'])
                # #view_used_mem()

                print("\n   Evaluating predictive ability...")
                acc_score_val, f1_score_val, metrics_val = discriminative_ability(model, datasets['val'],
                                                                                  name=str(run_ID), folder=folder,
                                                                                  mode='val')
                # #view_used_mem()

                acc_score_train, f1_score_train, metrics_train = discriminative_ability(model, datasets['train'],
                                                                                        name=str(run_ID),
                                                                                        folder=folder, mode='train')
                # #view_used_mem()

                print("\n   Evaluating generative ability...")
                generative_acc_val = generative_ability(model, clf, datasets['val'])
                # #view_used_mem()

                # Update log file
                update_info = '{},{},'.format(exp_ID, run_ID) + ''.join(
                    str(config[key]) + ',' for key in params_columns) + "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    float(recon_acc_val_set),
                    f1_score_val,
                    float(recon_acc_train_set),
                    f1_score_train,
                    generative_acc_val,
                    train_elbo,
                    val_elbo,
                    epoch,
                    config['max_epoch'],
                    tot_time,
                    str(tag))
                lock_filename = os.path.join(exp_folder, exp_class, 'access.lock')
                update_resource(log_filepath=os.path.join(exp_folder, exp_class, exp_class + '.csv'),
                                update_info=update_info, lock_filename=lock_filename)

                # Generate reconstruction and generation samples
                print('\nCreating reconstruction and generation samples...')
                image_generation(model, str(run_ID), folder=folder, n_samples=8)

                # view_used_mem()
                conditional_image_generation(model, str(run_ID), test_set=datasets['val'], folder=folder,
                                             img_suff="_val")
                # view_used_mem()
                conditional_image_generation(model, str(run_ID), test_set=datasets['test'], folder=folder,
                                             img_suff="_test")
                # view_used_mem()
                image_reconstruction(model, str(run_ID), folder=folder, img_suff="_val", img_dim=(3, 3, 3),
                                     test_set=datasets['val'])
                # view_used_mem()
                image_reconstruction(model, str(run_ID), folder=folder, img_suff="_test", img_dim=(3, 3, 3),
                                     test_set=datasets['test'])
                # view_used_mem()

            # Draw training and validation curves
            print('\nDrawing Learning Curves...')
            learning_curve(os.path.join(folder, str(run_ID)),
                           name=['train_info.npy', 'validation_info.npy'],
                           folder_path=os.path.join(folder, str(run_ID), 'learning_curve'), save=True, overwrite=True)
            # view_used_mem()
	    
            #os.remove(checkpoint_path)

            counter -= 1
            tot_number_exp += 1
            elapsed = time() - start_exp
            print('Done.')

    print("{} experiment(s) completed (total time:{})".format(tot_number_exp, elapsed))
