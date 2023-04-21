import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import LocationBasedGenerator
from data_loader import SimData, sim_collate_fn



def load_model_weights(weights_path, device="cpu"):
    """Load saved weights into model"""
    model = LocationBasedGenerator()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.to(device)
    return model


def visualize_original_and_reconstructed(start_batch, reconstructed_batch, file_suffix, save_dir=None):
    fig = plt.figure(figsize=(15, 15))
    for i in range(start_batch.size(0)):
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Input Image")
        plt.imshow(np.transpose(start_batch[i], (1, 2, 0)))
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Reconstructed Image")
        plt.imshow(np.transpose(reconstructed_batch[i], (1, 2, 0)))

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            image_path = f"{save_dir}/inference_{file_suffix}_{i}.png"
            plt.savefig(image_path, transparent=True, bbox_inches='tight')
            print("saved to", image_path)
        else:
            plt.show()
        plt.close(fig)


def inference(num_to_reconstruct: int, data_dir: str, weights_path: str, visualize=True, save_dir: str=None, device="cpu", random=False):
    """Run visual module using the given set of trained network weights. This function will visualize and
    Args:
        num_to_reconstruct (int) : the number of images to pass through the model and reconstruct
        data_dir (str) : directory containing the data split from which visualization images will be selected
        weights_path (str) : path to the set of trained model weights
        visualize (bool) : whether to visualize the input and reconstructed image pairs
        save_dir (str) : folder to save the reconstructed images
        device (str)
        random (bool) : whether to select images from data_dir randomly
    Returns:
        None
    """

    # Load model weights
    model = load_model_weights(weights_path, device='cpu')
    model.eval()

    eval_data = SimData(root_dir=data_dir, nb_samples=num_to_reconstruct, train_size=1.0)
    eval_loader = DataLoader(eval_data, batch_size=num_to_reconstruct, shuffle=random, collate_fn=sim_collate_fn)

    for batch in eval_loader:
        start, default, _ = [tensor.to(device) for tensor in batch[:3]]

        # Pass through model
        reconstructed = model(start, default, None, only_pred=True)

        # Combine original and reconstructed image
        start_stacked = torch.sum(start, dim=1).detach().cpu()
        reconstructed_stacked = torch.sum(reconstructed, dim=1).detach().cpu()

        if visualize:
            file_suffix = data_dir.split("/")[-1] + "_" + weights_path.split("/")[-1]
            visualize_original_and_reconstructed(start_stacked, reconstructed_stacked, file_suffix, save_dir)

    return


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-to-reconstruct",
        default=3,
        type=int,
        help="Number of images to pass through the model and reconstruct.",
    )
    parser.add_argument(
        "--select-randomly",
        default=False,
        type=bool,
        help="Whether to choose images randomly when visualizing"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="GPU with format 'cuda:[device number]' or 'cpu'",
    )
    parser.add_argument(
        "--data-dir",
        default="/data/workspace/16.412-learning-for-planning/data/sim_data/6objs_seg",
        type=str,
        help="Directory of the data split to load",
    )
    parser.add_argument(
        "--weights-path",
        default="/data/workspace/16.412-learning-for-planning/pre_models/model-sim-20230322-213256-5objs_seg",
        type=str,
        help="Trained model weights to load and use for inference"
    )
    parser.add_argument(
        "--visualize",
        default=True,
        type=bool,
        help="Whether to visualize the input and reconstructed image pairs"
    )
    parser.add_argument(
        "--save-path",
        default="/data/workspace/16.412-learning-for-planning/figures/reconstructed",
        type=str,
        help="Directory to save the reconstructed images"
    )
    parsed_args = parser.parse_args([])
    return parsed_args


if __name__ == "__main__":
    # Default args will load the model trained on 5 objects and visualizes reconstructions of 6-object scenes
    args = parse_arguments()
    inference(
        args.num_to_reconstruct,
        args.data_dir,
        args.weights_path,
        visualize=args.visualize,
        save_dir=args.save_path,
        device=args.device,
        random=args.select_randomly,
    )
