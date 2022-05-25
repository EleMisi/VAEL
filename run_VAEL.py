import argparse
from utils.mnist_utils.mnist_task_VAEL import run_mnist_vael
from utils.mario_utils.mario_task_VAEL import run_mario_vael

import config
import const_define as cd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run VAEL experiments')
    parser.add_argument('--task', type=str,
                        help='Experiment class', default='mario')

    args = parser.parse_args()

    if args.task == 'mario':
        config_object = config.mario_vael
        # Experiments and data folders
        exp_class = config_object['experiment_name']
        exp_folder = f'{cd.PROJECT_DIR}/experiments/'
        data_folder = f'{cd.PROJECT_DIR}/data/mario/'

        # Run experiments
        run_mario_vael(config_object['exp_config'], exp_class, exp_folder, data_folder,
                       batch_size=config_object['batch_size'], task=config_object['exp_config']['task'][0],
                       tag=config_object['exp_config']['tag'][0], device=config_object['device'], time_limit=100800,
                       early_stopping_info=config_object['early_stopping_info'], time_delta=300)

    elif args.task == 'mnist':
        config_object = config.mnist_vael
        # Experiments and data folders
        exp_class = config_object['experiment_name']
        exp_folder = f'{cd.PROJECT_DIR}/experiments/'
        data_folder = f'{cd.PROJECT_DIR}/data/MNIST/'

        # Run experiments
        run_mnist_vael(config_object['exp_config'], exp_class, exp_folder, data_folder, n_digits=config_object['n_digits'],
                       dataset_dimensions=config_object['dataset_dimensions'],
                       batch_size=config_object['batch_size'], task=config_object['exp_config']['task'][0],
                       tag=config_object['exp_config']['tag'][0], device=config_object['device'], time_limit=100800,
                       early_stopping_info=config_object['early_stopping_info'], time_delta=300)