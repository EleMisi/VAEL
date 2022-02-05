"""
VAEL experiment(s) configuration
"""
device = 'cpu'
n_digits = 10
exp_config= {'task': ['base'],
              'tag': [f'base_{n_digits}Digits'],
              'rec_loss': ['LAPLACE'],
              'max_epoch': [50],
              'n_exp': [1],
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