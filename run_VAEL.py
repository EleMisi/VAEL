import config
from utils.addition_task_VAEL import run_vael

if __name__ == "__main__":
    # Experiments and data folders
    exp_class = 'VAEL_' + config.exp_config['rec_loss'][0]
    exp_folder = './experiments/'
    data_folder = "./data/MNIST/2mnist_{}digits/".format(config.n_digits)
    data_file = "2mnist_{}digits.pt".format(config.n_digits)

    # Run experiments
    run_vael(config.exp_config, exp_class, exp_folder, data_folder, data_file, n_digits=config.n_digits,
             batch_size=config.batch_size, dataset_dimension=config.dataset_dimensions,
             task=config.exp_config['task'][0], tag=config.exp_config['tag'][0], device=config.device,
             time_limit=100800, early_stopping_info=config.early_stopping_info, classes=None, time_delta=300)
