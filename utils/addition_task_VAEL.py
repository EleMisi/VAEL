import os.path
import random
from datetime import datetime
from itertools import product
from pathlib import Path
from time import time, sleep

import numpy as np
import pandas as pd
import torch
from problog.formula import LogicFormula, LogicDAG
from problog.sdd_formula import SDD
from sklearn.model_selection import ParameterGrid
from torch import nn, optim

from models.vael import MNISTPairsVAELModel
from models.vael_network import Encoder, Decoder, MLP
from utils.nMNIST_addition import nMNIST, check_dataset
from utils.metrics_VAEL import reconstructive_ability, discriminative_ability, generative_ability
from utils.plot_utils_VAEL import conditional_image_generation, learning_curve, image_reconstruction, image_generation
from utils.problog_model import create_facts, define_ProbLog_model
from utils.train import train_PLVAE


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


def load_data(n_digits, sequence_len, batch_size, data_path, data_folder, task='base', tag='base', classes=None):

    if task == 'base':
        print("\n Base task\n")
        train_idxs = torch.load(data_folder + "train_indexes.pt")
        val_idxs = torch.load(data_folder + "val_indexes.pt")
        test_idxs = torch.load(data_folder + 'test_indexes.pt')

        # Prepare data
        train_set = nMNIST(sequence_len, worlds=None, digits=n_digits, batch_size=batch_size['train'], idxs=train_idxs,
                           train=True, sup=False, sup_digits=None, data_path=data_path)
        val_set = nMNIST(sequence_len, worlds=None, digits=n_digits, batch_size=batch_size['val'], idxs=val_idxs, train=True,
                         sup=False, sup_digits=None, data_path=data_path)
        test_set = nMNIST(sequence_len, worlds=None, digits=n_digits, batch_size=batch_size['test'], idxs=test_idxs,
                          train=False, sup=False, sup_digits=None, data_path=data_path)


    print("Train set: {} batches of {} images ({} images)".format(len(train_set), batch_size['train'],
                                                                  len(train_set) * batch_size['train']))
    print("Validation set: {} batches of {} images ({} images)".format(len(val_set), batch_size['val'],
                                                                  len(val_set) * batch_size['val']))
    print("Test set: {} batches of {} images ({} images)".format(len(test_set), batch_size['test'],
                                                                  len(test_set) * batch_size['test']))
    return train_set, val_set, test_set


def load_mnist_classifier(checkpoint_path, device):
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    clf = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))

    if torch.cuda.is_available():
        clf.load_state_dict(torch.load(checkpoint_path))
        clf = clf.to(device)
    else:
        clf.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    return clf


def define_experiment(exp_folder, exp_class, params, exp_counter):
    log_file = Path(os.path.join(exp_folder, exp_class, exp_class + '.csv'))
    params_columns = ['latent_dim_sub', 'latent_dim_sym', 'learning_rate', 'dropout', 'dropout_ENC', 'dropout_DEC',
                      'recon_w',
                      'kl_w',
                      'query_w', 'sup_w']
    if log_file.is_file():
        # Load file
        log_csv = pd.read_csv(os.path.join(exp_folder, exp_class, exp_class + '.csv'))

        # Check if the required number of test has been already satisfied
        required_exp = params['n_exp']
        exp_ID = ''.join(f'{params[key]}-' for key in params_columns)[:-1]
        if len(log_csv) > 0:
            n_exp = len(log_csv[log_csv['exp_ID'] == exp_ID]['run_ID'])

            if n_exp < required_exp:
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
                run_ID = None
        else:
            counter = required_exp
            run_ID = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
            print("\n\n0 compatible experiments found in file {} -> {} experiments to run.".format(
                exp_folder + exp_class + '.csv',
                counter))





    else:
        counter = params['n_exp']
        # Create log file
        log_file = open(os.path.join(exp_folder, exp_class, exp_class + '.csv'), 'w')
        header = 'exp_ID,run_ID,' + ''.join(str(key) + ',' for key in params_columns) + params[
            'rec_loss'] + "_recon_val,acc_discr_val," + params[
                     'rec_loss'] + "_recon_test,acc_discr_test,acc_gen,epochs,max_epoch,time,tag\n"
        log_file.write(header)
        # Define experiment ID
        exp_ID = ''.join(f'{params[key]}-' for key in params_columns)[:-1]
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

    return run_ID, exp_ID, exp_counter, counter


def build_model_dict(sequence_len, n_digits):
    """Define dictionary of pre-compiled ProbLog models"""
    possible_query_add = {2: list(range(0, (n_digits - 1) * 2 + 1))}
    rules = "addition(X,N) :- digit(X,1,N1), digit(X,2,N2), N is N1 + N2.\ndigits(X,Y):-digit(img,1,X), digit(img,2,Y)."
    facts = create_facts(sequence_len, n_digits=n_digits)
    model_dict = {'query': {add: "EMPTY" for add in possible_query_add[sequence_len]},
                  'evidence': {add: "EMPTY" for add in possible_query_add[sequence_len]}}

    for mode in ['query', 'evidence']:
        for add in model_dict[mode]:
            problog_model = define_ProbLog_model(facts, rules, label=add, digit_query='digits(X,Y)', mode=mode)
            lf = LogicFormula.create_from(problog_model)
            dag = LogicDAG.create_from(lf)
            sdd = SDD.create_from(dag)
            model_dict[mode][add] = sdd

    return model_dict


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


def run_vael(param_grid, exp_class, exp_folder, data_folder, data_file, n_digits, batch_size, dataset_dimension, task='base', tag='base', device='cpu',
             time_limit=500, early_stopping_info=None, classes=None, time_delta=350):
    print("\nDevice:", device)
    print()

    # Load data
    sequence_len = 2  # Number of digits in the sequence
    label_dim = n_digits * sequence_len

    # Load data
    data_path = os.path.join(data_folder, data_file)
    # Check whether dataset exists, if not build it
    check_dataset(n_digits, data_folder, data_file, dataset_dimension)
    train_set, val_set, test_set = load_data(n_digits=n_digits, sequence_len=sequence_len, batch_size=batch_size,
                                             data_path=data_path, data_folder=data_folder, task=task, tag=tag,
                                             classes=classes)

    # MNIST classifier
    clf = load_mnist_classifier(checkpoint_path='./utils/MNIST_classifier.pt', device=device)
    clf.eval()

    # Define pre-compiled ProbLog programs and worlds-queries matrix
    model_dict = build_model_dict(sequence_len, n_digits)
    w_q = build_worlds_queries_matrix(sequence_len, n_digits)
    w_q = w_q.to(device)

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
            run_ID, exp_ID, exp_counter, counter = define_experiment(exp_folder, exp_class, config, exp_counter)
            if not counter:
                break

            # Build VAEL model
            encoder = Encoder(hidden_channels=64, latent_dim=config['latent_dim_sym'] + config['latent_dim_sub'],
                              dropout=config['dropout_ENC'])
            decoder = Decoder(label_dim=label_dim, hidden_channels=64, latent_dim=config['latent_dim_sub'],
                              dropout=config['dropout_DEC'])
            mlp = MLP(in_features=config['latent_dim_sym'],
                      n_digits=n_digits)
            model = MNISTPairsVAELModel(encoder=encoder, decoder=decoder, mlp=mlp,
                                        latent_dims=(config['latent_dim_sym'], config['latent_dim_sub']),
                                        model_dict=model_dict, w_q=w_q, dropout=config['dropout'], is_train=True,
                                        device=device)
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

            # Reset data generator
            train_set.reset_counter()
            val_set.reset_counter()

            # Timing
            start = time()

            # Train
            checkpoint_path, epoch, train_info, validation_info = train_PLVAE(model,
                                                                              optimizer,
                                                                              n_epochs=config['max_epoch'],
                                                                              train_set=train_set,
                                                                              val_set=val_set,
                                                                              early_stopping_info=early_stopping_info,
                                                                              run_ID=str(run_ID),
                                                                              recon_w=config['recon_w'],
                                                                              kl_w=config['kl_w'],
                                                                              query_w=config['query_w'],
                                                                              sup_w=config['sup_w'],
                                                                              folder=os.path.join(exp_folder, exp_class,
                                                                                                  exp_ID),
                                                                              rec_loss=config['rec_loss'],
                                                                              train_batch_size=batch_size['train'],
                                                                              val_batch_size=batch_size['val'])

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

            # Reset generator
            train_set.reset_counter()
            val_set.reset_counter()

            # Evaluation
            model.eval()
            with torch.no_grad():
                print("\n\n****TEST MODEL WITH CONFIGURATION*****\n", config)

                print("\n   Evaluating reconstructive ability on test set...")
                test_set.reset_counter()
                recon_acc_test_set = reconstructive_ability(model, test_set, config['rec_loss'])
                print("   Evaluating reconstructive ability on validation set...")
                val_set.reset_counter()
                recon_acc_val_set = reconstructive_ability(model, val_set, config['rec_loss'])
                print("Number Of Images Tested =", test_set.samples_x_world * len(test_set))
                print("Reconstructive Accuracy = {} ({})".format(float(recon_acc_test_set), config['rec_loss']))
                print("\n   Evaluating predictive ability on test set...")
                test_set.reset_counter()
                predictive_acc_test_set = discriminative_ability(model, test_set)
                print("   Evaluating predictive ability on validation set...")
                val_set.reset_counter()
                predictive_acc_val_set = discriminative_ability(model, val_set)
                print("Number Of Images Tested =", test_set.samples_x_world * len(test_set))
                print("Discriminative Accuracy =", predictive_acc_test_set)
                print("\n   Evaluating generative...")
                test_set.reset_counter()
                n_sample = 100
                generative_acc = generative_ability(model, clf, n_sample)
                print("Number Of Images Tested =", n_sample * len(test_set))
                print("Generative Accuracy =", generative_acc)

            # Update log file
            params_columns = ['latent_dim_sub', 'latent_dim_sym', 'learning_rate', 'dropout', 'dropout_DEC', 'recon_w',
                              'kl_w', 'query_w', 'sup_w']
            update_info = '{},{},'.format(exp_ID, run_ID) + ''.join(
                str(config[key]) + ',' for key in params_columns) + "{},{},{},{},{},{},{},{},{}\n".format(
                float(recon_acc_val_set),
                predictive_acc_val_set,
                float(recon_acc_test_set),
                predictive_acc_test_set,
                generative_acc,
                epoch,
                config['max_epoch'],
                tot_time,
                str(tag))
            lock_filename = os.path.join(exp_folder, exp_class, 'access.lock')
            update_resource(log_filepath=os.path.join(exp_folder, exp_class, exp_class + '.csv'),
                            update_info=update_info, lock_filename=lock_filename)

            # Generate reconstruction and generation samples
            folder = os.path.join(exp_folder, exp_class, exp_ID)
            print('\nCreating reconstruction and generation samples...')
            test_set.reset_counter()
            image_generation(model, str(run_ID), folder=folder)
            test_set.reset_counter()
            image_reconstruction(model, str(run_ID), folder=folder, img_suff="", img_dim=[28, 56], test_set=test_set)
            test_set.reset_counter()
            conditional_image_generation(model, str(run_ID), folder=folder)

            # Draw training and validation curves
            print('\nDrawing Learning Curves...')
            learning_curve(os.path.join(folder, str(run_ID)),
                           name=['train_info.npy', 'validation_info.npy'],
                           folder_path=os.path.join(folder, str(run_ID), 'learning_curve'), save=True, overwrite=True)

            counter -= 1
            tot_number_exp += 1
            elapsed = time() - start_exp
            print('Done.')

    print("{} experiment(s) completed (total time:{})".format(tot_number_exp, elapsed))
