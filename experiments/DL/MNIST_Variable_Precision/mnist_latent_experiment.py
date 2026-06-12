"""MNIST latent space variability experiment with PRISM Stochastic Rounding.

Runs inference with a pre-trained model and captures latent representations
at each layer. Designed to be run multiple times under SR to observe
numerical variability in the latent space.

Usage:
    python3 mnist_latent_experiment.py \\
        --model-path /mnist/model/mnist_cnn.pt \\
        --output-dir /mnist/results/precision_24 \\
        --num-runs 30
"""

import argparse
import os
import pickle

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from common import Net


def run_inference_with_embeddings(model, device, test_loader, output_dir, run_id):
    """Run inference on the first batch, capture all layer activations."""
    model.eval()

    # Storage for intermediate activations
    activations = {}

    # Register forward hooks to capture activations
    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = output.detach().cpu()
        return hook_fn

    hooks = []
    hooks.append(model.conv1.register_forward_hook(make_hook("conv1")))
    hooks.append(model.conv2.register_forward_hook(make_hook("conv2")))
    hooks.append(model.fc1.register_forward_hook(make_hook("fc1")))
    hooks.append(model.fc2.register_forward_hook(make_hook("fc2")))

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Compute metrics
            batch_loss = F.nll_loss(output, target, reduction="sum").item()
            test_loss += batch_loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Save activations and predictions for first batch only
            activations["output"] = output.detach().cpu()
            activations["pred"] = pred.detach().cpu()
            activations["target"] = target.detach().cpu()
            activations["loss"] = batch_loss / data.size(0)
            break  # First batch only (1000 samples)

    # Remove hooks
    for h in hooks:
        h.remove()

    n_samples = activations["target"].shape[0]  # actual batch size processed
    test_loss /= n_samples
    accuracy = 100.0 * correct / n_samples

    print(
        f"Run {run_id}: loss={activations['loss']:.6f}, "
        f"accuracy={correct}/{n_samples} ({accuracy:.1f}%)"
    )

    # Save all activations
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"activations_run_{run_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(activations, f)

    return activations


def main():
    parser = argparse.ArgumentParser(
        description="MNIST Latent Space Variability Experiment"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to pre-trained mnist_cnn.pt",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save activations (e.g. /mnist/results/precision_24)",
    )
    parser.add_argument(
        "--run-id", type=int, default=1,
        help="Run identifier (1..N)",
    )
    parser.add_argument(
        "--num-runs", type=int, default=1,
        help="Number of runs to execute",
    )
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Random seed (fixed for reproducible data loading)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_dataset = datasets.MNIST(
        args.data_dir, train=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size
    )

    # Load pre-trained model
    model = Net().to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )

    # Run inference and capture embeddings
    if args.num_runs > 1:
        for r in range(1, args.num_runs + 1):
            run_inference_with_embeddings(
                model, device, test_loader, args.output_dir, r
            )
    else:
        run_inference_with_embeddings(
            model, device, test_loader, args.output_dir, args.run_id
        )


if __name__ == "__main__":
    main()
