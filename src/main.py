from argparse import ArgumentParser

import torch.utils.data

from data import read_all_data, all_devices
from federated_util import *
from test_hparams import test_hyperparameters
from unsupervised_data import get_client_unsupervised_initial_splitting


def main(setup: str, collaborative: bool):
    Ctp.set_automatic_skip(True)
    Ctp.print('\n\t\t\t\t\t' + "FEDAVG" + ' ' + setup.upper() + ' ' + "AUTOENCODER"
              + ' TESTING\n', bold=True)

    common_params = {'n_features': 115,
                     'normalization': 'min-max',  # "min-max", "0-mean 1var"
                     'test_bs': 4096,
                     'p_test': 0.2,
                     'p_unused': 0.01,  # Proportion of data left unused between *train_val set* and *test set*
                     'val_part': None,
                     # This is the proportion of *train_val set* that goes into the validation set, not the proportion of all data
                     'n_splits': 5,  # number of splits in the cross validation
                     'n_random_reruns': 1,
                     'cuda': False,  # It looks like cuda is slower than CPU for me so I enforce using the CPU
                     'benign_prop': 0.0787,
                     # Desired proportion of benign data in the train/validation sets (or None to keep the natural proportions)
                     'samples_per_device': 100_000}  # Total number of datapoints (train & val + unused + test) for each device.

    # p_test, p_unused and p_train_val are the proportions of *all data* that go into respectively the *test set*, the *unused set*
    # and the *train_val set*.
    # val_part and threshold_part are the proportions of the the *train_val set* used for respectively the validation and the threshold
    # note that we either use one or the other: when grid searching we do not compute the threshold, so we have the *train_val set*
    # split between val_part proportion of validation data and (1. - val_part) proportion of train data
    # when testing hyper-params, we have *train & validation set* split between threshold_part proportion of threshold data and
    # (1. - threshold_part) proportion of train data.
    # benign_prop is yet another thing, determining the proportion of benign data when applicable (everywhere except in the *train_val set*
    # of the unsupervised method)

    p_train_val = 1. - common_params['p_test'] - common_params['p_unused']

    if common_params['val_part'] is None:
        val_part = 1. / common_params['n_splits']
    else:
        val_part = common_params['val_part']

    common_params.update({'p_train_val': p_train_val, 'val_part': val_part})

    if common_params['cuda']:
        Ctp.print('Using CUDA')
    else:
        Ctp.print('Using CPU')

    autoencoder_params = {'activation_fn': torch.nn.ELU,
                          'threshold_part': 0.5,
                          'quantile': 0.95,
                          'epochs': 120,
                          'train_bs': 64,
                          'optimizer': torch.optim.SGD,
                          'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                          'lr_scheduler_params': {'step_size': 20, 'gamma': 0.5}}

    # Note that other architecture-specific parameters, such as the dimensions of the hidden layers, can be specified in either in the
    # varying_params for the grid searches, or in the configurations_params for the tests.

    n_devices = len(all_devices)
    fedavg_params = {'federation_rounds': 1,
                     'gamma_round': 0.75}

    federation_params = {'aggregation_function': federated_averaging,
                         'resampling': None}  # s-resampling


    federation_params.update(fedavg_params)
    Ctp.print("Federation params: {}".format(federation_params), color='blue')

    poisoning_params = {'n_malicious': 0,
                        'data_poisoning': None,
                        'p_poison': None,
                        'model_update_factor': 1.0,
                        'model_poisoning': None}

    if poisoning_params['n_malicious'] != 0:
        Ctp.print("Poisoning params: {}".format(poisoning_params), color='red')

    # 9 configurations in which we have 8 clients (each one owns the data from 1 device) and the data from the last device is left unseen.
    # decentralized_configurations = [{'clients_devices': [[i] for i in range(n_devices) if i != test_device],
    #                                  'test_devices': [test_device]} for test_device in range(n_devices)]
    test_device = 0  # pick one
    decentralized_configurations = [{
        'clients_devices': [[i] for i in range(n_devices) if i != test_device],
        'test_devices': [test_device]
    }]

    local_configurations = [{'clients_devices': [[known_device]],
                             'test_devices': [i for i in range(n_devices) if i != known_device]}
                            for known_device in range(n_devices)]

    # 9 configurations in which we have 1 client (owning the data from 8 devices) and he data from the last device is left unseen.
    centralized_configurations = [{'clients_devices': [[i for i in range(n_devices) if i != test_device]],
                                   'test_devices': [test_device]} for test_device in range(n_devices)]

    if setup == 'centralized':
        configurations = centralized_configurations
    elif setup == 'decentralized':
        if collaborative:
            configurations = decentralized_configurations
        else:
            configurations = local_configurations
    else:
        raise ValueError

    Ctp.print(configurations)

    # Loading the data
    all_data = read_all_data()

    constant_params = {**common_params, **autoencoder_params}
    splitting_function = get_client_unsupervised_initial_splitting
    constant_params.update(federation_params)

    # set the hyper-parameters specific to each configuration (overrides the parameters defined in constant_params)
    configurations_params = [{'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 1e-05}},
                                {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}}]

    test_hyperparameters(all_data, setup, splitting_function, constant_params, configurations_params, configurations)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('setup', help='centralized or decentralized')
    parser.add_argument('experiment', help='Experiment to run (classifier or autoencoder)')

    # Grid-search removed — only test mode is supported
    parser.add_argument('--test', dest='test', action='store_true',
                        help='Run final training and evaluation (default)')
    parser.set_defaults(test=True)

    collaborative_parser = parser.add_mutually_exclusive_group(required=False)
    collaborative_parser.add_argument('--collaborative', dest='collaborative', action='store_true',
                                      help='Makes the clients collaborate by sharing validation results (results will be per configuration).')
    collaborative_parser.add_argument('--no-collaborative', dest='collaborative', action='store_false',
                                      help='Makes the clients not collaborate; results will be per client.')
    parser.set_defaults(collaborative=True)

    federated_parser = parser.add_mutually_exclusive_group(required=False)
    federated_parser.add_argument('--fedavg', dest='federated', action='store_const',
                                  const='fedavg', help='Federation of the models (default: None)')
    parser.set_defaults(federated='fedavg')

    verbose_parser = parser.add_mutually_exclusive_group(required=False)
    verbose_parser.add_argument('--verbose', dest='verbose', action='store_true')
    verbose_parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    parser.add_argument('--verbose-depth', dest='max_depth', type=int, help='Maximum number of nested sections after which the printing will stop')
    parser.set_defaults(max_depth=None)

    args = parser.parse_args()

    if not args.verbose:  # Deactivate all printing in the console
        Ctp.deactivate()

    if args.max_depth is not None:
        Ctp.set_max_depth(args.max_depth)  # Set the max depth at which we print in the console

    main(args.setup, args.collaborative)
