import os
import shutil
import tempfile
import argparse
import datetime
import json
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
import pytorch_ssim
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from data_loaders_dict_mrpet import get_loader

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def main():

    parser = argparse.ArgumentParser(description='MONAI 3D translation')
    parser.add_argument('-d', '--dataset', default='spdp_fbp', type=str)
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='number of samples in each batch')
    parser.add_argument('-m', '--mode', default='concat', type=str)
    parser.add_argument('-nl', '--noiseloss', default='l1', type=str)
    parser.add_argument('-il', '--imageloss', default='l2', type=str)
    parser.add_argument('--msssim', action='store_true')
    parser.add_argument('--weighted', action='store_true')
    parser.add_argument('--alpha', default=0.8, type=float, metavar='N',
                        help='alpha coeff for L1 and MS-SSIM loss')
    
    args = parser.parse_args()
    print(args)

    timestamp = datetime.datetime.now().strftime("%m%d%y%H%M%S")
    training_folder = f"./{args.dataset}_sp2dp_spmrcond_{timestamp}"

    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
        os.makedirs(f"{training_folder}/weights/DiffusionModelUNet")
    else:
        print(f"{training_folder} exists!")
        exit()

    with open(f'{training_folder}/params.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ae_sp = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=(False, False, True),
    ).to(device)
    ae_sp_path = '../checkpoints/compression_models/ae_kl_spfbp/49.ckpt'
    ae_sp.load_state_dict(torch.load(ae_sp_path, map_location=device))

    ae_dp = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=(False, False, True),
    ).to(device)
    ae_dp_path = '../checkpoints/compression_models/ae_kl_dpfbp/49.ckpt'
    ae_dp.load_state_dict(torch.load(ae_dp_path, map_location=device))

    ae_mr = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=(False, False, True),
    ).to(device)
    ae_mr_path = '../checkpoints/compression_models/ae_kl_mr/45.ckpt'
    ae_mr.load_state_dict(torch.load(ae_mr_path, map_location=device))

    print("Results saved at: ", training_folder)

    dataset_train = get_loader(f'./{args.dataset}/train', batch_size=args.batch_size, mode="train")

    #--------------------------------------------------------------------------------------------------------
    if args.mode == "concat":
        my_in_channels = 9
    else:
        my_in_channels = 3

    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=my_in_channels,
        out_channels=3,
        num_res_blocks=1,
        num_channels=(32, 64, 64),
        attention_levels=(False, True, True),
        num_head_channels=(0, 64, 64),
        with_conditioning=True,
        cross_attention_dim=64,
    )
    unet.to(device)
    print("DiffusionModelUNet parameters:", get_n_params(unet))

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)

    check_data = first(dataset_train)
    with torch.no_grad():
        with autocast(enabled=True):
            z = ae_dp.encode_stage_2_inputs(check_data["trgt"].to(device))

    print(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1 / torch.std(z)

    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)

    n_epochs = args.epochs
    epoch_loss_list = []
    ae_sp.eval()
    ae_dp.eval()
    scaler = GradScaler()

    first_batch = first(dataset_train)
    z = ae_sp.encode_stage_2_inputs(first_batch["img"].to(device))

    for epoch in range(n_epochs):
        unet.train()
        epoch_loss, n_loss, il1_loss, ims_loss, i_loss = 0, 0, 0, 0, 0
        progress_bar = tqdm(enumerate(dataset_train), total=len(dataset_train), ncols=150)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in progress_bar:
            sp_images = batch["img"].to(device)
            dp_images = batch["trgt"].to(device)
            mr_images = batch["mr"].to(device)
            
            with torch.no_grad():
                sp_latent = ae_sp.encode_stage_2_inputs(sp_images) * scale_factor
                mr_latent = ae_mr.encode_stage_2_inputs(mr_images) * scale_factor
                spmr_latent = torch.cat((sp_latent, mr_latent), 1)
                
                if args.mode == "crossattn":
                    sp_latent = torch.reshape(sp_latent, (sp_latent.shape[0], 1, -1))
            
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # Generate random noise
                noise = torch.randn_like(z).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (dp_images.shape[0],), device=dp_images.device
                ).long()

                # Get model prediction
                # print(spmr_latent.shape)

                dp_latent, noise_pred = inferer(
                    inputs=dp_images, 
                    autoencoder_model=ae_dp, 
                    diffusion_model=unet, 
                    noise=noise, 
                    timesteps=timesteps, 
                    condition=spmr_latent, 
                    mode=args.mode
                )

                predicted_prev_sample, predicted_orig_latent = inferer.sample(
                    input_noise=noise, 
                    autoencoder_model=ae_dp, 
                    diffusion_model=unet, 
                    scheduler=scheduler, 
                    conditioning=spmr_latent, 
                    mode='concat', 
                    verbose=False,
                    jay_boolean=True)

                if args.noiseloss == "l2":
                    noise_loss = F.mse_loss(noise_pred.float(), noise.float())
                elif args.noiseloss == "l1":
                    noise_loss = F.l1_loss(noise_pred.float(), noise.float())

                if args.imageloss == "l2":
                    image_l1_loss = F.mse_loss(predicted_orig_latent, dp_latent)
                    image_ssim_loss = 1 - pytorch_ssim.msssim_3d(predicted_orig_latent, dp_latent, normalize=True)
                    image_loss = args.alpha*image_ssim_loss + (1 - args.alpha)*image_l1_loss

                elif args.imageloss == "l1":
                    image_l1_loss = F.l1_loss(predicted_orig_latent, dp_latent)
                    image_ssim_loss = 1 - pytorch_ssim.msssim_3d(predicted_orig_latent, dp_latent, normalize=True)
                    image_loss = args.alpha*image_ssim_loss + (1 - args.alpha)*image_l1_loss

                loss = noise_loss + image_loss
                # loss = noise_loss + image_l1_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            epoch_loss += loss.item()
            n_loss += noise_loss.item()
            il1_loss += image_l1_loss.item()
            ims_loss += image_ssim_loss.item()
            i_loss += image_loss.item()

            progress_bar.set_postfix({
                "total_loss": epoch_loss / (step + 1), 
                "noise_loss": n_loss / (step + 1), 
                "image_l1": il1_loss / (step + 1),
                "image_ms": ims_loss / (step + 1),
                "image_loss": i_loss / (step + 1)
                })
        epoch_loss_list.append(epoch_loss / (step + 1))
        
        mpath = os.path.join(f"{training_folder}/weights/DiffusionModelUNet", '{}.ckpt'.format(epoch+1))
        torch.save(unet.state_dict(), mpath)

    plt.plot(epoch_loss_list)
    plt.title("Learning Curves", fontsize=20)
    plt.plot(epoch_loss_list)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(f"{args.dataset}_sp2dp_diffunet_learning_curves.png")

if __name__ == '__main__':

    print("DiffusionModelUNet: SPFBP to DPFBP with SP+MR condition")

    main()