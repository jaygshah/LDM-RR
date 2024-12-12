import os
import shutil
import tempfile
import argparse
import datetime
import json
import time
import numpy as np

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
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from data_loaders_dict import get_loader

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def main():

    parser = argparse.ArgumentParser(description='LDM-RR AutoencoderKL')
    parser.add_argument('-d', '--dataset', default='spdp_fbp', type=str)
    parser.add_argument('-mod', '--modality', default='sp', type=str)
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='number of samples in each batch')
    
    args = parser.parse_args()
    print(args)

    timestamp = datetime.datetime.now().strftime("%m%d%y%H%M%S")
    training_folder = f"./{args.dataset}_{args.modality}_{timestamp}"

    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
        os.makedirs(f"{training_folder}/weights/AutoencoderKL")
    else:
        print(f"{training_folder} exists!")
        exit()

    with open(f'{training_folder}/params.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=(False, False, True),
    )
    autoencoder.to(device)
    print("AutoencoderKL parameters:", get_n_params(autoencoder))

    discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
    discriminator.to(device)

    print("Results saved at: ", training_folder)

    dataset_train = get_loader(f'./{args.dataset}/train', batch_size=args.batch_size, mode="train")
    
    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)


    def KL_loss(z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]

    adv_weight = 0.01
    perceptual_weight = 0.001
    kl_weight = 1e-6

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

    n_epochs = args.epochs
    autoencoder_warm_up_n_epochs = 5
    val_interval = 10
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    val_recon_epoch_loss_list = []
    intermediary_images = []
    n_example_images = 4

    for epoch in range(n_epochs):
        autoencoder.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        progress_bar = tqdm(enumerate(dataset_train), total=len(dataset_train), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            if args.modality == "sp":
                images = batch["img"].to(device)
            elif args.modality == "dp":
                images = batch["trgt"].to(device)
            elif args.modality == "mr":
                images = batch["mrs"].to(device)
            else:
                print(f"{args.modality} is not supported. Please select from [sp, dp, mr]")
                break

            # Generator part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)
            kl_loss = KL_loss(z_mu, z_sigma)

            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            epoch_loss += recons_loss.item()
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

            progress_bar.set_postfix(
                {
                    "recons_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                }
            )
        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
        epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

        mpath = os.path.join(f"{training_folder}/weights/AutoencoderKL", '{}.ckpt'.format(epoch+1))
        torch.save(autoencoder.state_dict(), mpath)

    del discriminator
    del loss_perceptual
    torch.cuda.empty_cache()

    plt.style.use("ggplot")
    plt.title("Learning Curves", fontsize=20)
    plt.plot(epoch_recon_loss_list)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(f"{args.dataset}_{timestamp}_autoencoder_learning_curves.png")
    plt.clf()

    plt.title("Adversarial Training Curves", fontsize=20)
    plt.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
    plt.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(f"{args.dataset}_{timestamp}_{args.modality}_autoencoder_adv_training_curves.png")
    plt.clf()

if __name__ == '__main__':

    print("AutoencoderKL Training...")

    main()