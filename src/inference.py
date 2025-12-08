import argparse
import json
import os
import time
from memory_profiler import profile
from ptflops import get_model_complexity_info

import numpy as np
import pandas as pd
import torch

from architectures import SimpleAutoencoder, NormalizingModel, Threshold
from federated_util import init_federated_models, model_aggregation

def load_params(constants_path: str, configs_path: str, config_idx: int):
    """Load constant and configuration-specific params from JSON files."""
    with open(constants_path, "r") as f:
        constant_params = json.load(f)

    with open(configs_path, "r") as f:
        configs = json.load(f)

    # configs is expected to be a list of dicts
    if isinstance(configs, dict):
        configs_list = list(configs.values())
    else:
        configs_list = configs

    if not (0 <= config_idx < len(configs_list)):
        raise IndexError(f"config_idx {config_idx} is out of range (0..{len(configs_list)-1})")

    config_params = configs_list[config_idx]

    return constant_params, config_params


def load_model_and_threshold(
    model_path: str,
    threshold_path: str,
    constant_params: dict,
    config_params: dict,
    device: torch.device,
):
    """
    Load:
      - global model state_dict from model_path
      - global threshold (module or state_dict or tensor) from threshold_path

    model_path was saved with:
        torch.save(global_model.state_dict(), model_path)

    threshold_path was saved with either:
        torch.save(global_threshold.state_dict(), threshold_path)
    or:
        torch.save(global_threshold, threshold_path)
    or:
        torch.save(global_threshold.threshold, threshold_path)
    """
    # 1) Build empty model architecture matching training
    global_model = NormalizingModel(
        SimpleAutoencoder(
            activation_function=torch.nn.ELU,
            hidden_layers=config_params["hidden_layers"],
        ),
        sub=torch.zeros(constant_params["n_features"]),
        div=torch.ones(constant_params["n_features"]),
    )

    # 2) Load model state_dict
    state_dict = torch.load(model_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError("Expected model .pth file to contain a state_dict (dict-like).")
    global_model.load_state_dict(state_dict)
    global_model.to(device)
    global_model.eval()

    # 3) Load threshold object from file
    threshold_obj = torch.load(threshold_path, map_location=device)

    # Cases:
    #  - state_dict: {'threshold': tensor(...)}
    #  - Threshold module with .threshold
    #  - plain tensor or float
    if isinstance(threshold_obj, dict):
        if "threshold" in threshold_obj:
            val = threshold_obj["threshold"]
        else:
            # Take first value in dict
            val = next(iter(threshold_obj.values()))
        threshold_value = float(val)
    elif isinstance(threshold_obj, Threshold):
        threshold_value = float(threshold_obj.threshold.item())
    elif isinstance(threshold_obj, torch.nn.Module) and hasattr(threshold_obj, "threshold"):
        threshold_value = float(threshold_obj.threshold.item())
    elif isinstance(threshold_obj, torch.Tensor):
        threshold_value = float(threshold_obj.item())
    else:
        # e.g. saved as a plain float
        threshold_value = float(threshold_obj)

    return global_model, threshold_value


def load_data(data_path: str, has_label: bool):
    """Load data from CSV.

    If has_label=True, assumes the last column is the label.
    Returns:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray or None
    """
    df = pd.read_csv(data_path)

    if has_label:
        X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
        y = df.iloc[:, -1].to_numpy()
    else:
        X = df.to_numpy(dtype=np.float32)
        y = None

    return X, y

@profile
def run_inference(
    model,
    threshold_value: float,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
):
    """Run anomaly detection using reconstruction error > threshold.

    Returns:
        errors: np.ndarray of shape (n_samples,)
        preds: np.ndarray of shape (n_samples,), 1 = anomaly, 0 = normal
    """
    model.to(device)

    n = X.shape[0]
    errors_list = []
    preds_list = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = torch.from_numpy(X[start:end]).to(device)

            recon = model(batch)
            # Mean squared error per sample
            batch_errors = torch.mean((batch - recon) ** 2, dim=1)

            batch_preds = (batch_errors > threshold_value).long()

            errors_list.append(batch_errors.cpu().numpy())
            preds_list.append(batch_preds.cpu().numpy())

    errors = np.concatenate(errors_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)

    return errors, preds


def main():
    parser = argparse.ArgumentParser(description="Inference with global autoencoder model + threshold")

    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the global model state_dict (.pth)")
    parser.add_argument("--threshold-path", type=str, required=True,
                        help="Path to the global threshold state_dict (.pth)")
    parser.add_argument("--constants-path", type=str, required=True,
                        help="Path to constant_params.json")
    parser.add_argument("--configs-path", type=str, required=True,
                        help="Path to configurations_params.json")
    parser.add_argument("--config-idx", type=int, default=0,
                        help="Index of configuration to use (0-based)")

    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to CSV file with data for inference")
    parser.add_argument("--has-label", action="store_true",
                        help="If set, assume last column in CSV is the label")

    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run on: 'cpu' or 'cuda' if available")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size for inference")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Optional path to save predictions as CSV")

    args = parser.parse_args()

    # Device
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Load params
    constant_params, config_params = load_params(
        args.constants_path,
        args.configs_path,
        args.config_idx,
    )
    print("Loaded constant_params and config_params.")
    print("constant_params keys:", list(constant_params.keys()))
    print("config_params keys:", list(config_params.keys()))

    # Load model + threshold (using state_dicts)
    global_model, threshold_value = load_model_and_threshold(
        args.model_path,
        args.threshold_path,
        constant_params,
        config_params,
        device,
    )
    # global_model = NormalizingModel(SimpleAutoencoder(activation_function=torch.nn.ELU, hidden_layers=config_params["hidden_layers"]),
    #                                 sub=torch.zeros(constant_params["n_features"]), div=torch.ones(constant_params["n_features"]))
    # global_threshold = Threshold(torch.tensor(0.))
    print(f"Loaded global model from {args.model_path}")
    print(f"Loaded threshold: {threshold_value:.6f}")

    # Load data
    X, y = load_data(args.data_path, has_label=args.has_label)
    print(f"Loaded data from {args.data_path} with shape {X.shape}")

    # Sanity check on feature count
    if "n_features" in constant_params:
        expected = constant_params["n_features"]
        if X.shape[1] != expected:
            print(
                f"WARNING: Data has {X.shape[1]} features but n_features in constant_params is {expected}."
            )

    # Assuming your input is a flat vector of size n_features
    n_features = constant_params["n_features"]

    with torch.cuda.device(device) if device.type == "cuda" else torch.device("cpu"):
        macs, params = get_model_complexity_info(
            global_model,
            (n_features,),       # single sample, shape (n_features,)
            as_strings=False,
            verbose=False,
        )

    print(f"Model params: {params:.0f}")
    print(f"MACs per sample (approx): {macs:.0f}")
    print(f"FLOPs per sample (approx): {2 * macs:.0f}")

    # Run inference
    t0 = time.perf_counter()
    errors, preds = run_inference(
        global_model,
        threshold_value,
        X,
        device,
        batch_size=args.batch_size,
    )
    t1 = time.perf_counter()

    print("\nInference completed.")
    print(f"Total inference time: {t1 - t0:.4f} seconds")
    print(f"Throughput: {len(X) / (t1 - t0):.2f} samples/second")

    print(f"\nThreshold: {threshold_value:.6f}")
    print(f"Mean reconstruction error: {errors.mean():.6f}")
    print(f"Predicted anomalies: {preds.sum()} / {len(preds)} samples")

    # If labels are available, compute a quick accuracy (assuming 1=anomaly, 0=normal)
    if y is not None:
        unique_labels = set(np.unique(y))
        if unique_labels <= {0, 1}:
            acc = (preds == y).mean()
            print(f"Label-based accuracy: {acc*100:.2f}%")
        else:
            print(f"Labels are not binary {0,1} (got {unique_labels}); skipping accuracy.")

    # Optionally save output
    if args.output_path is not None:
        out_df = pd.DataFrame({
            "reconstruction_error": errors,
            "prediction": preds,
        })
        if y is not None:
            out_df["label"] = y

        if args.output_path.strip():
            dir_name = os.path.dirname(args.output_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

        out_df.to_csv(args.output_path, index=False)
        print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
