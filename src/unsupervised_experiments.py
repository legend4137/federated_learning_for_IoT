from types import SimpleNamespace
from typing import Tuple, List, Dict

import os
import torch
from context_printer import Color
from context_printer import ContextPrinter as Ctp
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from architectures import SimpleAutoencoder, NormalizingModel, Threshold
from data import device_names, ClientData, FederationData, get_benign_attack_samples_per_device
from federated_util import init_federated_models, model_aggregation, select_mimicked_client, model_poisoning
from metrics import BinaryClassificationResult
from print_util import print_federation_round
from unsupervised_data import prepare_dataloaders
from unsupervised_ml import multitrain_autoencoders, multitest_autoencoders, compute_thresholds, train_autoencoder


def federated_thresholds(models: List[torch.nn.Module], threshold_dls: List[DataLoader], global_threshold: torch.nn.Module,
                         params: SimpleNamespace, global_thresholds: List[float]) -> None:
    # Computation of the thresholds
    thresholds = compute_thresholds(opts=list(zip(['Computing threshold for client {} on: '.format(i) + device_names(client_devices)
                                                   for i, client_devices in enumerate(params.clients_devices)], threshold_dls, models)),
                                    quantile=params.quantile,
                                    main_title='Computing the thresholds', color=Color.DARK_PURPLE)

    # Aggregation of the thresholds
    global_threshold, thresholds = model_aggregation(global_threshold, thresholds, params, verbose=True)
    Ctp.print('Global threshold: {:.6f}'.format(global_threshold.threshold.item()))
    global_thresholds.append(global_threshold.threshold.item())


def federated_testing(global_model: torch.nn.Module, global_threshold: torch.nn.Module,
                      local_test_dls_dicts: List[Dict[str, DataLoader]], new_test_dls_dict: Dict[str, DataLoader],
                      params: SimpleNamespace, local_results: List[BinaryClassificationResult],
                      new_devices_results: List[BinaryClassificationResult]) -> None:

    # Global model testing on each client's data
    tests = []
    for client_id, client_devices in enumerate(params.clients_devices):
        tests.append(('Testing global model on: ' + device_names(client_devices), local_test_dls_dicts[client_id], global_model, global_threshold))

    local_results.append(multitest_autoencoders(tests=tests,
                                                main_title='Testing the global model on data from all clients', color=Color.BLUE))

    # Global model testing on new devices
    new_devices_results.append(multitest_autoencoders(tests=list(zip(['Testing global model on: ' + device_names(params.test_devices)],
                                                                     [new_test_dls_dict], [global_model], [global_threshold])),
                                                      main_title='Testing the global model on the new devices: ' + device_names(
                                                          params.test_devices),
                                                      color=Color.DARK_CYAN))


def fedavg_autoencoders_train_test(train_val_data: FederationData, local_test_data: FederationData,
                                   new_test_data: ClientData, params: SimpleNamespace)\
        -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult], List[float]]:
    # Preparation of the dataloaders
    train_dls, threshold_dls, local_test_dls_dicts, new_test_dls_dict = prepare_dataloaders(train_val_data, local_test_data, new_test_data, params)

    # Initialization of the models
    global_model, models = init_federated_models(train_dls, params, architecture=SimpleAutoencoder)
    global_threshold = Threshold(torch.tensor(0.))

    # Initialization of the results
    local_results, new_devices_results, global_thresholds = [], [], []

    # Selection of a client to mimic in case we use the mimic attack
    mimicked_client_id = select_mimicked_client(params)

    for federation_round in range(params.federation_rounds):
        print_federation_round(federation_round, params.federation_rounds)

        # Local training of each client
        multitrain_autoencoders(trains=list(zip(['Training client {} on: '.format(i) + device_names(client_devices)
                                                 for i, client_devices in enumerate(params.clients_devices)],
                                                train_dls, models)),
                                params=params, lr_factor=(params.gamma_round ** federation_round),
                                main_title='Training the clients', color=Color.GREEN)
        
        # Model poisoning attacks
        models = model_poisoning(global_model, models, params, mimicked_client_id=mimicked_client_id, verbose=True)

        # Aggregation
        global_model, models = model_aggregation(global_model, models, params, verbose=True)

        # Compute and aggregate thresholds
        federated_thresholds(models, threshold_dls, global_threshold, params, global_thresholds)

        # Testing
        federated_testing(global_model, global_threshold, local_test_dls_dicts, new_test_dls_dict, params, local_results, new_devices_results)

        Ctp.exit_section()

    # --- SAVE GLOBAL MODEL AFTER TRAINING ---
    # save_dir = params.output_dir if hasattr(params, 'output_dir') else "saved_models"
    # os.makedirs(save_dir, exist_ok=True)

    # model_path = os.path.join(save_dir, "global_model_final.pth")
    # threshold_path = os.path.join(save_dir, "global_threshold_final.pth")

    # torch.save(global_model.state_dict(), model_path)
    # torch.save(global_threshold.state_dict(), threshold_path)

    # Ctp.print(f"Saved final global model to: {model_path}", color=Color.YELLOW)
    # Ctp.print(f"Saved final global threshold to: {threshold_path}", color=Color.YELLOW)
    # ----------------------------------------

    return local_results, new_devices_results, global_thresholds, global_model, global_threshold


