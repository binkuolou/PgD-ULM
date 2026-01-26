import argparse
import os
import numpy as np
import torch
import scipy.io as sio
from conditional_DDPM import DDPM, ContextUnet
from DDIM import DiffusionProcessDDIM
from torchvision.utils import make_grid, save_image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate microbubble PSFs using Physical Parameter-Guided Diffusion Model')

    # Model parameters
    parser.add_argument('--model_path', type=str, default='models/best_model_Interaction.pth',
                        help='Path to pre-trained model weights')
    parser.add_argument('--n_feat', type=int, default=128,
                        help='Number of features in the U-Net model')
    parser.add_argument('--n_classes', type=int, default=4,
                        help='Number of conditional imaging parameters')
    parser.add_argument('--n_T', type=int, default=500,
                        help='Number of diffusion timesteps')

    # Generation parameters
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of PSFs to generate')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for generation')
    parser.add_argument('--sampling_method', type=str, choices=['DDIM', 'DDPM'], default='DDIM',
                        help='Sampling method to use (DDIM or DDPM)')
    parser.add_argument('--tau', type=int, default=10,
                        help='Number of DDIM steps (only for DDIM sampling)')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta parameter (0.0 for deterministic sampling)')
    parser.add_argument('--guide_w', type=float, default=0.0,
                        help='Classifier-free guidance weight')

    # Physical parameters
    parser.add_argument('--frequency', type=float, default=7.35,
                        help='Imaging frequency in MHz')
    parser.add_argument('--pitch', type=int, default=240,
                        help='Transducer element pitch in micrometers')
    parser.add_argument('--elements', type=int, default=128,
                        help='Number of active transducer elements')
    parser.add_argument('--pulses', type=int, default=2,
                        help='Number of transmitted half pulses')

    # Output parameters
    parser.add_argument('--save_path', type=str, default='outputs/',
                        help='Directory to save generated PSFs')
    parser.add_argument('--save_format', type=str, choices=['mat', 'npy', 'png'], default='mat',
                        help='Format to save generated PSFs (mat, npy, or png)')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to run generation on')

    return parser.parse_args()


def load_model(args):
    """Load pre-trained diffusion model"""
    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=1, n_feat=args.n_feat, n_classes=args.n_classes),
        betas=(1e-4, 0.02),
        n_T=args.n_T,
        device=args.device,
        drop_prob=0.01
    )

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}. Please download pre-trained weights.")

    state_dict = torch.load(args.model_path, map_location=args.device)
    ddpm.load_state_dict(state_dict, strict=True)
    ddpm.to(args.device)
    ddpm.eval()

    return ddpm


def normalize_condition(c):
    """Normalize condition parameters using fixed min-max values"""
    c_min = torch.tensor([5, 100, 64, 2]).view(-1)
    c_range = torch.tensor([10.625, 200, 64, 4]).view(-1)
    return (c - c_min) / c_range


def generate_psfs_ddim(ddpm, c, args):
    """Generate PSFs using DDIM sampling"""
    ddim = DiffusionProcessDDIM(
        beta_1=1e-4,
        beta_T=0.02,
        T=args.n_T,
        c=c,
        w=args.guide_w,
        diffusion_fn=ddpm,
        device=args.device,
        shape=(1, 48, 48),
        eta=args.eta,
        tau=args.tau
    )

    total_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    all_psfs = []

    for batch_idx in range(total_batches):
        batch_size = min(args.batch_size, args.num_samples - len(all_psfs))
        x_gen = ddim.sampling(batch_size, only_final=True)
        all_psfs.append(x_gen.cpu().numpy())

        print(f"Generated batch {batch_idx + 1}/{total_batches} ({len(all_psfs) * batch_size}/{args.num_samples} PSFs)")

    return np.concatenate(all_psfs, axis=0)[:args.num_samples]


def generate_psfs_ddpm(ddpm, c, args):
    """Generate PSFs using DDPM sampling"""
    total_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    all_psfs = []

    for batch_idx in range(total_batches):
        batch_size = min(args.batch_size, args.num_samples - len(all_psfs))
        x_gen, _ = ddpm.sample(batch_size, c, (1, 48, 48), args.device, guide_w=args.guide_w)
        all_psfs.append(x_gen.cpu().numpy())

        print(f"Generated batch {batch_idx + 1}/{total_batches} ({len(all_psfs) * batch_size}/{args.num_samples} PSFs)")

    return np.concatenate(all_psfs, axis=0)[:args.num_samples]


def save_results(psfs, args):
    """Save generated PSFs in specified format"""
    os.makedirs(args.save_path, exist_ok=True)

    if args.save_format == 'mat':
        save_name = os.path.join(args.save_path, f'PSF_{args.sampling_method}_w{args.guide_w}_tau{args.tau}.mat')
        sio.savemat(save_name, {'PSF': psfs})
        print(f"Saved {len(psfs)} PSFs to {save_name}")


    elif args.save_format == 'png':
        # Save individual PSFs as PNG images
        for i, psf in enumerate(psfs):
            save_name = os.path.join(args.save_path, f'PSF_{args.sampling_method}_w{args.guide_w}_{i:04d}.png')
            psf_tensor = torch.from_numpy(psf).unsqueeze(0)
            save_image(psf_tensor, save_name, normalize=True, value_range=(-1, 1))

        print(f"Saved {len(psfs)} PSFs as PNG images to {args.save_path}")


def main():
    args = parse_args()
    print(f"Starting PSF generation with {args.sampling_method} sampling...")
    print(
        f"Target parameters: Frequency={args.frequency}MHz, Pitch={args.pitch}μm, Elements={args.elements}, Pulses={args.pulses}")
    print(f"Generating {args.num_samples} PSFs with batch size {args.batch_size}")

    # Load model
    ddpm = load_model(args)

    # Prepare condition parameters
    c = torch.tensor([args.frequency, args.pitch, args.elements, args.pulses]).view(-1)
    c = normalize_condition(c).to(args.device)

    # Generate PSFs
    if args.sampling_method == 'DDIM':
        psfs = generate_psfs_ddim(ddpm, c, args)
    else:  # DDPM
        psfs = generate_psfs_ddpm(ddpm, c, args)

    # Save results
    save_results(psfs, args)

    print("Generation completed successfully!")


if __name__ == '__main__':
    main()