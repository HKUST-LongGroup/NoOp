import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from diffusers import StableDiffusionPipeline, AutoencoderKL, EulerDiscreteScheduler
from datasets import load_dataset
from PIL import Image
from torchvision import transforms as tvt
from typing import Union, Tuple, Optional
from multiprocessing import Process, Value
import math
import sys
import csv
import torch.fft as fft

def ratio_high_freq(noise, cutoff_ratio=0.3):
    assert noise.ndim == 4 and noise.shape[2] == noise.shape[3],
    spec   = fft.rfft2(noise, norm='ortho')
    spec   = fft.fftshift(spec, dim=(-2, -1))
    power  = spec.real.pow(2) + spec.imag.pow(2)    

    H, W   = power.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    yy     = yy - H // 2
    xx     = xx - W // 2
    radius = torch.sqrt(xx**2 + yy**2).to(power)
    r_max  = radius.max()

    mask_low  = radius <= cutoff_ratio * r_max
    mask_high = ~mask_low

    E_low  = power[..., mask_low].sum()
    E_high = power[..., mask_high].sum()

    return (E_high / (E_low + E_high)).item()


# Meta-Networ
class Noises(nn.Module):
    def __init__(self):
        super(Noises, self).__init__()
        self.noises = nn.Parameter(torch.randn(1000, 4, 64, 64))
    def forward(self, t):
        return self.noises[t].unsqueeze(0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = DoubleConv(4, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        
        self.outc = nn.Conv2d(64, 4, kernel_size=1)
        nn.init.zeros_(self.outc.weight)   
        nn.init.zeros_(self.outc.bias)     

    def forward(self, x):
        x1 = self.inc(x)    # [1, 64, 64, 64]
        x2 = self.down1(x1) # [1, 128, 32, 32]
        x3 = self.down2(x2) # [1, 256, 16, 16]
        x4 = self.down3(x3) # [1, 512, 8, 8]
        
        x = self.up1(x4, x3) # [1, 256, 16, 16]
        x = self.up2(x, x2)  # [1, 128, 32, 32]
        x = self.up3(x, x1)  # [1, 64, 64, 64]
        
        return self.outc(x) # [1, 4, 64, 64]
    

class NonUniformDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle

        self.total_size = len(self.dataset)
        self.base_samples_per_worker = self.total_size // self.num_replicas
        self.remainder = self.total_size % self.num_replicas

        if rank < self.remainder:
            self.num_samples = self.base_samples_per_worker + 1
            self.offset = rank * (self.base_samples_per_worker + 1)
        else:
            self.num_samples = self.base_samples_per_worker
            self.offset = self.remainder * (self.base_samples_per_worker + 1) + (rank - self.remainder) * self.base_samples_per_worker

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        local_indices = indices[self.offset : self.offset + self.num_samples]
        
        return iter(local_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


@torch.no_grad()
def load_image(pil_img, target_size):
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  


# Custom Dataset that returns a PIL image and label
class InferenceDataset(Dataset):
    def __init__(self, ds, img_placeholder):
        self.ds = ds
        self.img_placeholder = img_placeholder
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        sample = self.ds[idx]
        # Return the tensor and label directly
        return load_image(sample[self.img_placeholder].convert('RGB'),512).squeeze(0), sample['label']


# Few-shot dataset
class FewShotDatasetWrapper(Dataset):
    def __init__(self, data):
        self.data = data  
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


@torch.no_grad()
def img_to_latents(x, vae):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


def z_score_normalize(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor, unbiased=False)
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor


def train_epoch(diffusion_model, train_loader, optimizer, device, ddp_noises, seed, rank, epoch, prompts, unet_input_prompt_size, if_l1, ddp_unet, t_values, if_image_specific):

    seed = seed + epoch * 1000 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    total_loss = 0.0

    pipeline = StableDiffusionPipeline.from_pretrained(diffusion_model).to(device)
    pipeline.scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model, subfolder="scheduler")

    ddp_noises.train()

    for param in pipeline.unet.parameters():
        param.requires_grad = False
    for param in pipeline.vae.parameters():
        param.requires_grad = False
    for param in pipeline.text_encoder.parameters():
        param.requires_grad = False

    inputs = pipeline.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    text_embeddings = pipeline.text_encoder(inputs.input_ids.to(device)).last_hidden_state.detach()
    for batch in train_loader:
        optimizer.zero_grad()
        t = random.choice(t_values)
        noise = ddp_noises(t)
        timesteps = torch.tensor([t], device=device)
        x0, label = batch
        label = label[0].item()  
        x0 = x0.squeeze(0).to(device)
        noise_offset = ddp_unet(x0)
        other_indices = np.random.choice([i for i in range(len(prompts)) if i != label], size=unet_input_prompt_size-1, replace=False)
        selected_indices = [label] + other_indices.tolist()
        random.shuffle(selected_indices)  
        selected_text_embeddings = text_embeddings[selected_indices]  # (unet_input_prompt_size, seq_len, emb_dim)

        refined_noise = noise + noise_offset if if_image_specific else noise

        if "rec" in diffusion_model:
            x_t = (t+1)/1000 * refined_noise + (1 - (t+1)/1000) * x0
            v_gt = x0 - refined_noise
        else:
            x_t = x0 * (pipeline.scheduler.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                refined_noise * ((1 - pipeline.scheduler.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1).to(device)
            v_gt = refined_noise

        noisy_latents = torch.cat([x_t]*unet_input_prompt_size, dim=0)
        v_gts = torch.cat([v_gt]*unet_input_prompt_size, dim=0)

        predicted_v = pipeline.unet(noisy_latents, timesteps, 
                                    encoder_hidden_states=selected_text_embeddings).sample

        if if_l1:
            errors = F.l1_loss(predicted_v, v_gts, reduction='none').mean(dim=(1,2,3))
        else:
            errors = F.mse_loss(predicted_v, v_gts, reduction='none').mean(dim=(1,2,3))
        norm_errors = z_score_normalize(errors)
        logits = -norm_errors

        target_in_subset = selected_indices.index(label)
        target = torch.tensor([target_in_subset], device=device)
        loss = F.cross_entropy(logits.unsqueeze(0), target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    total_loss = torch.tensor(total_loss).to(device)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    return total_loss.item() / (len(train_loader.dataset) * dist.get_world_size())


@torch.no_grad()
def inference(diffusion_model, test_loader, device, seed, rank, prompts, prompt_bz, if_l1, ddp_noises, ddp_unet, t_values, if_image_specific):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pipeline = StableDiffusionPipeline.from_pretrained(diffusion_model).to(device)
    pipeline.scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model, subfolder="scheduler")

    correct = 0
    total = 0

    inputs = pipeline.tokenizer(prompts, return_tensors="pt", padding="max_length")
    text_embeddings = pipeline.text_encoder(inputs.input_ids.to(device)).last_hidden_state

    for test_idx, (input_img, label) in enumerate(test_loader):
        input_img = input_img.to(device)
        x0 = img_to_latents(input_img, pipeline.vae)
        noise_offset = ddp_unet(x0)
        current_candidates = list(range(len(prompts)))
        ddp_noises.eval()
        k = 1
        total_errors = torch.zeros(len(current_candidates)).to(device)
        current_candidates_dict = {p: idx for idx, p in enumerate(current_candidates)}

        t_dict = {}
        for each_t in t_values:
            t_dict[each_t] = torch.zeros(len(current_candidates)).to(device)

        combinations = [(p, t) for p in current_candidates for t in t_values]

        num_batches = (len(combinations) + prompt_bz - 1) // prompt_bz
        
        for batch_idx in range(num_batches):
            start = batch_idx * prompt_bz
            end = min(start + prompt_bz, len(combinations))
            batch_comb = combinations[start:end]
            
            batch_t = list({t for _, t in batch_comb})
            t_noise_dict = {}
            
            for t in batch_t:
                refined_noise = ddp_noises(t) + noise_offset if if_image_specific else ddp_noises(t)
                if "rec" in diffusion_model:
                    x_t = (t+1)/1000 * refined_noise + (1 - (t+1)/1000) * x0
                    v_gt = x0 - refined_noise
                    t_noise_dict[t] = (x_t, v_gt)
                else:
                    x_t = x0 * (pipeline.scheduler.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                        refined_noise * ((1 - pipeline.scheduler.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1).to(device)
                    v_gt = refined_noise
                    t_noise_dict[t] = (x_t, v_gt)

            batch_p = [p for p, _ in batch_comb]
            batch_t = [t for _, t in batch_comb]
            
            batch_text_emb = text_embeddings[batch_p]
            batch_x_t = torch.cat([t_noise_dict[t][0] for t in batch_t], dim=0)
            batch_v_gt = torch.cat([t_noise_dict[t][1] for t in batch_t], dim=0)
            batch_timesteps = torch.tensor(batch_t, device=device)
            
            predicted_v = pipeline.unet(batch_x_t, batch_timesteps, 
                                        encoder_hidden_states=batch_text_emb).sample
            if if_l1:
                batch_errors = F.l1_loss(predicted_v, batch_v_gt, reduction='none').mean(dim=(1,2,3)) 
            else:
                batch_errors = F.mse_loss(predicted_v, batch_v_gt, reduction='none').mean(dim=(1,2,3)) 


            for idx, (p, t) in enumerate(batch_comb):
                candidate_idx = current_candidates_dict[p]
                t_dict[t][candidate_idx] = batch_errors[idx]

        for each_t in t_values:
            total_errors += t_dict[each_t]


        sorted_indices = torch.argsort(total_errors)
        current_candidates = [current_candidates[i] for i in sorted_indices[:k]]

        pred_index = current_candidates[0]
        correct += (1 if label.item() == pred_index else 0)
        total += 1

    correct = torch.tensor(correct).to(device)
    total = torch.tensor(total).to(device)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return correct.item(), total.item()



parser = argparse.ArgumentParser(description='Noise Tuning')
parser.add_argument('--dataset',
                    type=str,
                    default='cifar10')
parser.add_argument('--seed',
                    type=int,
                    default=2025)
parser.add_argument('--infer_freq',
                    type=int,
                    default=5)
parser.add_argument('--shot',
                    type=int,
                    default=8)
parser.add_argument('--prompt_bz',
                    type=int,
                    default=100)


def main():
    dist.init_process_group(backend='gloo')
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
        
    args = parser.parse_args()
    dataset_name = args.dataset
    prompt_bz = args.prompt_bz
    shot = args.shot
    diffusion_model = "stabilityai/stable-diffusion-2-base"
    t_values = [500]
    epochs = 20

    log_path = 'results_14/'+dataset_name+'_'+str(shot)+'_'+str(t_values[0])+'.log'
    sys.stdout = open(log_path, 'a')

    seed = args.seed + rank  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False    



    if dataset_name == 'pets':
        inference_dataset_url = "Multimodal-Fatima/OxfordPets_test"
        training_dataset_url = "my_datasets/pets_" + str(shot) + "shot.pt"
        place_holder = 'image'
        unet_input_prompt_size = 15
        if_l1 = True
        prompts = ['a photo of a Abyssinian cat, a type of pet.', 'a photo of a American Bulldog, a type of pet.',
                   'a photo of a American Pit Bull Terrier dog, a type of pet.', 'a photo of a Basset Hound dog, a type of pet.',
                   'a photo of a Beagle dog, a type of pet.', 'a photo of a Bengal cat, a type of pet.',
                   'a photo of a Birman cat, a type of pet.', 'a photo of a Bombay cat, a type of pet.',
                   'a photo of a Boxer dog, a type of pet.', 'a photo of a British Shorthair cat, a type of pet.',
                   'a photo of a Chihuahua dog, a type of pet.', 'a photo of a Egyptian Mau cat, a type of pet.',
                   'a photo of a English Cocker Spaniel dog, a type of pet.', 'a photo of a English Setter dog, a type of pet.',
                   'a photo of a German Shorthaired dog, a type of pet.', 'a photo of a Great Pyrenees dog, a type of pet.',
                   'a photo of a Havanese dog, a type of pet.', 'a photo of a Japanese Chin dog, a type of pet.',
                   'a photo of a Keeshond dog, a type of pet.', 'a photo of a Leonberger dog, a type of pet.',
                   'a photo of a Maine Coon cat, a type of pet.', 'a photo of a Miniature Pinscher dog, a type of pet.',
                   'a photo of a Newfoundland dog, a type of pet.', 'a photo of a Persian cat, a type of pet.',
                   'a photo of a Pomeranian dog, a type of pet.', 'a photo of a Pug dog, a type of pet.',
                   'a photo of a Ragdoll cat, a type of pet.', 'a photo of a Russian Blue cat, a type of pet.',
                   'a photo of a Saint Bernard dog, a type of pet.', 'a photo of a Samoyed dog, a type of pet.',
                   'a photo of a Scottish Terrier dog, a type of pet.', 'a photo of a Shiba Inu dog, a type of pet.',
                   'a photo of a Siamese cat, a type of pet.', 'a photo of a Sphynx cat, a type of pet.',
                   'a photo of a Staffordshire Bull Terrier dog, a type of pet.', 'a photo of a Wheaten Terrier dog, a type of pet.',
                   'a photo of a Yorkshire Terrier dog, a type of pet.'
        ]
    if dataset_name == 'cifar10':
        inference_dataset_url = "uoft-cs/cifar10"
        training_dataset_url = "my_datasets/cifar10_" + str(shot) + "shot.pt"
        place_holder = 'img'
        unet_input_prompt_size = 10
        if_l1 = True
        prompts = [
                "a blurry photo of an airplane.", "a blurry photo of an automobile.",
                "a blurry photo of a bird.", "a blurry photo of a cat.",
                "a blurry photo of a deer.", "a blurry photo of a dog.",
                "a blurry photo of a frog.", "a blurry photo of a horse.",
                "a blurry photo of a ship.", "a blurry photo of a truck."
            ]
    if dataset_name == 'aircraft':
        inference_dataset_url = "Multimodal-Fatima/FGVC_Aircraft_test"
        training_dataset_url = "my_datasets/aircraft_" + str(shot) + "shot.pt"
        place_holder = 'image'
        unet_input_prompt_size = 15
        if_l1 = False
        prompts = ['a photo of a 707-320, a type of aircraft.', 'a photo of a 727-200, a type of aircraft.', 'a photo of a 737-200, a type of aircraft.', 'a photo of a 737-300, a type of aircraft.', 'a photo of a 737-400, a type of aircraft.', 'a photo of a 737-500, a type of aircraft.', 'a photo of a 737-600, a type of aircraft.', 'a photo of a 737-700, a type of aircraft.', 'a photo of a 737-800, a type of aircraft.', 'a photo of a 737-900, a type of aircraft.', 'a photo of a 747-100, a type of aircraft.', 'a photo of a 747-200, a type of aircraft.', 'a photo of a 747-300, a type of aircraft.', 'a photo of a 747-400, a type of aircraft.', 'a photo of a 757-200, a type of aircraft.', 'a photo of a 757-300, a type of aircraft.', 'a photo of a 767-200, a type of aircraft.', 'a photo of a 767-300, a type of aircraft.', 'a photo of a 767-400, a type of aircraft.', 'a photo of a 777-200, a type of aircraft.', 'a photo of a 777-300, a type of aircraft.', 'a photo of a A300B4, a type of aircraft.', 'a photo of a A310, a type of aircraft.', 'a photo of a A318, a type of aircraft.', 'a photo of a A319, a type of aircraft.', 'a photo of a A320, a type of aircraft.', 'a photo of a A321, a type of aircraft.', 'a photo of a A330-200, a type of aircraft.', 'a photo of a A330-300, a type of aircraft.', 'a photo of a A340-200, a type of aircraft.', 'a photo of a A340-300, a type of aircraft.', 'a photo of a A340-500, a type of aircraft.', 'a photo of a A340-600, a type of aircraft.', 'a photo of a A380, a type of aircraft.', 'a photo of a ATR-42, a type of aircraft.', 'a photo of a ATR-72, a type of aircraft.', 'a photo of a An-12, a type of aircraft.', 'a photo of a BAE 146-200, a type of aircraft.', 'a photo of a BAE 146-300, a type of aircraft.', 'a photo of a BAE-125, a type of aircraft.', 'a photo of a Beechcraft 1900, a type of aircraft.', 'a photo of a Boeing 717, a type of aircraft.', 'a photo of a C-130, a type of aircraft.', 'a photo of a C-47, a type of aircraft.', 'a photo of a CRJ-200, a type of aircraft.', 'a photo of a CRJ-700, a type of aircraft.', 'a photo of a CRJ-900, a type of aircraft.', 'a photo of a Cessna 172, a type of aircraft.', 'a photo of a Cessna 208, a type of aircraft.', 'a photo of a Cessna 525, a type of aircraft.', 'a photo of a Cessna 560, a type of aircraft.', 'a photo of a Challenger 600, a type of aircraft.', 'a photo of a DC-10, a type of aircraft.', 'a photo of a DC-3, a type of aircraft.', 'a photo of a DC-6, a type of aircraft.', 'a photo of a DC-8, a type of aircraft.', 'a photo of a DC-9-30, a type of aircraft.', 'a photo of a DH-82, a type of aircraft.', 'a photo of a DHC-1, a type of aircraft.', 'a photo of a DHC-6, a type of aircraft.', 'a photo of a DHC-8-100, a type of aircraft.', 'a photo of a DHC-8-300, a type of aircraft.', 'a photo of a DR-400, a type of aircraft.', 'a photo of a Dornier 328, a type of aircraft.', 'a photo of a E-170, a type of aircraft.', 'a photo of a E-190, a type of aircraft.', 'a photo of a E-195, a type of aircraft.', 'a photo of a EMB-120, a type of aircraft.', 'a photo of a ERJ 135, a type of aircraft.', 'a photo of a ERJ 145, a type of aircraft.', 'a photo of a Embraer Legacy 600, a type of aircraft.', 'a photo of a Eurofighter Typhoon, a type of aircraft.', 'a photo of a F-16A/B, a type of aircraft.', 'a photo of a F/A-18, a type of aircraft.', 'a photo of a Falcon 2000, a type of aircraft.', 'a photo of a Falcon 900, a type of aircraft.', 'a photo of a Fokker 100, a type of aircraft.', 'a photo of a Fokker 50, a type of aircraft.', 'a photo of a Fokker 70, a type of aircraft.', 'a photo of a Global Express, a type of aircraft.', 'a photo of a Gulfstream IV, a type of aircraft.', 'a photo of a Gulfstream V, a type of aircraft.', 'a photo of a Hawk T1, a type of aircraft.', 'a photo of a Il-76, a type of aircraft.', 'a photo of a L-1011, a type of aircraft.', 'a photo of a MD-11, a type of aircraft.', 'a photo of a MD-80, a type of aircraft.', 'a photo of a MD-87, a type of aircraft.', 'a photo of a MD-90, a type of aircraft.', 'a photo of a Metroliner, a type of aircraft.', 'a photo of a Model B200, a type of aircraft.', 'a photo of a PA-28, a type of aircraft.', 'a photo of a SR-20, a type of aircraft.', 'a photo of a Saab 2000, a type of aircraft.', 'a photo of a Saab 340, a type of aircraft.', 'a photo of a Spitfire, a type of aircraft.', 'a photo of a Tornado, a type of aircraft.', 'a photo of a Tu-134, a type of aircraft.', 'a photo of a Tu-154, a type of aircraft.', 'a photo of a Yak-42, a type of aircraft.']
    if dataset_name == 'dtd':
        inference_dataset_url = "Multimodal-Fatima/DTD_parition1_test"
        training_dataset_url = "my_datasets/dtd_" + str(shot) + "shot.pt"
        place_holder = 'image'
        unet_input_prompt_size = 15
        if_l1 = True
        prompts = ['a photo of a banded texture.', 'a photo of a blotchy texture.', 'a photo of a braided texture.', 'a photo of a bubbly texture.', 'a photo of a bumpy texture.', 'a photo of a chequered texture.', 'a photo of a cobwebbed texture.', 'a photo of a cracked texture.', 'a photo of a crosshatched texture.', 'a photo of a crystalline texture.', 'a photo of a dotted texture.', 'a photo of a fibrous texture.', 'a photo of a flecked texture.', 'a photo of a freckled texture.', 'a photo of a frilly texture.', 'a photo of a gauzy texture.', 'a photo of a grid texture.', 'a photo of a grooved texture.', 'a photo of a honeycombed texture.', 'a photo of a interlaced texture.', 'a photo of a knitted texture.', 'a photo of a lacelike texture.', 'a photo of a lined texture.', 'a photo of a marbled texture.', 'a photo of a matted texture.', 'a photo of a meshed texture.', 'a photo of a paisley texture.', 'a photo of a perforated texture.', 'a photo of a pitted texture.', 'a photo of a pleated texture.', 'a photo of a polka-dotted texture.', 'a photo of a porous texture.', 'a photo of a potholed texture.', 'a photo of a scaly texture.', 'a photo of a smeared texture.', 'a photo of a spiralled texture.', 'a photo of a sprinkled texture.', 'a photo of a stained texture.', 'a photo of a stratified texture.', 'a photo of a striped texture.', 'a photo of a studded texture.', 'a photo of a swirly texture.', 'a photo of a veined texture.', 'a photo of a waffled texture.', 'a photo of a woven texture.', 'a photo of a wrinkled texture.', 'a photo of a zigzagged texture.']
    if dataset_name == 'stl10':
        inference_dataset_url = "tanganke/stl10"
        training_dataset_url = "my_datasets/stl10_" + str(shot) + "shot.pt"
        place_holder = 'image'
        unet_input_prompt_size = 10
        if_l1 = True
        prompts = ['a photo of a airplane.', 'a photo of a bird.', 'a photo of a car.', 'a photo of a cat.', 'a photo of a deer.', 'a photo of a dog.', 'a photo of a horse.', 'a photo of a monkey.', 'a photo of a ship.', 'a photo of a truck.']
    if dataset_name == 'eurosat':
        inference_dataset_url = "tanganke/eurosat"
        training_dataset_url = "my_datasets/eurosat_" + str(shot) + "shot.pt"
        place_holder = 'image'
        unet_input_prompt_size = 10
        if_l1 = True
        prompts = ["a centered satellite photo of a annual crop land.", "a centered satellite photo of a forest.", "a centered satellite photo of a brushland or shrubland.", "a centered satellite photo of a highway or road.", "a centered satellite photo of a industrial buildings or commercial buildings.", "a centered satellite photo of a pasture land.", "a centered satellite photo of a permanent crop land.", "a centered satellite photo of a residential buildings or homes or apartments.", "a centered satellite photo of a river.", "a centered satellite photo of a lake or sea."]
    if dataset_name == 'flowers':
        inference_dataset_url = "dpdl-benchmark/oxford_flowers102"
        training_dataset_url = "my_datasets/flowers_" + str(shot) + "shot.pt"
        place_holder = 'image'
        unet_input_prompt_size = 15
        if_l1 = True
        prompts = ['a photo of a pink primrose, a type of flower.', 'a photo of a hard-leaved pocket orchid, a type of flower.', 'a photo of a canterbury bells, a type of flower.', 'a photo of a sweet pea, a type of flower.', 'a photo of a english marigold, a type of flower.', 'a photo of a tiger lily, a type of flower.', 'a photo of a moon orchid, a type of flower.', 'a photo of a bird of paradise, a type of flower.', 'a photo of a monkshood, a type of flower.', 'a photo of a globe thistle, a type of flower.', 'a photo of a snapdragon, a type of flower.', "a photo of a colt's foot, a type of flower.", 'a photo of a king protea, a type of flower.', 'a photo of a spear thistle, a type of flower.', 'a photo of a yellow iris, a type of flower.', 'a photo of a globe flower, a type of flower.', 'a photo of a purple coneflower, a type of flower.', 'a photo of a peruvian lily, a type of flower.', 'a photo of a balloon flower, a type of flower.', 'a photo of a giant white arum lily, a type of flower.', 'a photo of a fire lily, a type of flower.', 'a photo of a pincushion flower, a type of flower.', 'a photo of a fritillary, a type of flower.', 'a photo of a red ginger, a type of flower.', 'a photo of a grape hyacinth, a type of flower.', 'a photo of a corn poppy, a type of flower.', 'a photo of a prince of wales feathers, a type of flower.', 'a photo of a stemless gentian, a type of flower.', 'a photo of a artichoke, a type of flower.', 'a photo of a sweet william, a type of flower.', 'a photo of a carnation, a type of flower.', 'a photo of a garden phlox, a type of flower.', 'a photo of a love in the mist, a type of flower.', 'a photo of a mexican aster, a type of flower.', 'a photo of a alpine sea holly, a type of flower.', 'a photo of a ruby-lipped cattleya, a type of flower.', 'a photo of a cape flower, a type of flower.', 'a photo of a great masterwort, a type of flower.', 'a photo of a siam tulip, a type of flower.', 'a photo of a lenten rose, a type of flower.', 'a photo of a barbeton daisy, a type of flower.', 'a photo of a daffodil, a type of flower.', 'a photo of a sword lily, a type of flower.', 'a photo of a poinsettia, a type of flower.', 'a photo of a bolero deep blue, a type of flower.', 'a photo of a wallflower, a type of flower.', 'a photo of a marigold, a type of flower.', 'a photo of a buttercup, a type of flower.', 'a photo of a oxeye daisy, a type of flower.', 'a photo of a common dandelion, a type of flower.', 'a photo of a petunia, a type of flower.', 'a photo of a wild pansy, a type of flower.', 'a photo of a primula, a type of flower.', 'a photo of a sunflower, a type of flower.', 'a photo of a pelargonium, a type of flower.', 'a photo of a bishop of llandaff, a type of flower.', 'a photo of a gaura, a type of flower.', 'a photo of a geranium, a type of flower.', 'a photo of a orange dahlia, a type of flower.', 'a photo of a pink and yellow dahlia, a type of flower.', 'a photo of a cautleya spicata, a type of flower.', 'a photo of a japanese anemone, a type of flower.', 'a photo of a black-eyed susan, a type of flower.', 'a photo of a silverbush, a type of flower.', 'a photo of a californian poppy, a type of flower.', 'a photo of a osteospermum, a type of flower.', 'a photo of a spring crocus, a type of flower.', 'a photo of a bearded iris, a type of flower.', 'a photo of a windflower, a type of flower.', 'a photo of a tree poppy, a type of flower.', 'a photo of a gazania, a type of flower.', 'a photo of a azalea, a type of flower.', 'a photo of a water lily, a type of flower.', 'a photo of a rose, a type of flower.', 'a photo of a thorn apple, a type of flower.', 'a photo of a morning glory, a type of flower.', 'a photo of a passion flower, a type of flower.', 'a photo of a lotus, a type of flower.', 'a photo of a toad lily, a type of flower.', 'a photo of a anthurium, a type of flower.', 'a photo of a frangipani, a type of flower.', 'a photo of a clematis, a type of flower.', 'a photo of a hibiscus, a type of flower.', 'a photo of a columbine, a type of flower.', 'a photo of a desert-rose, a type of flower.', 'a photo of a tree mallow, a type of flower.', 'a photo of a magnolia, a type of flower.', 'a photo of a cyclamen, a type of flower.', 'a photo of a watercress, a type of flower.', 'a photo of a canna lily, a type of flower.', 'a photo of a hippeastrum, a type of flower.', 'a photo of a bee balm, a type of flower.', 'a photo of a air plant, a type of flower.', 'a photo of a foxglove, a type of flower.', 'a photo of a bougainvillea, a type of flower.', 'a photo of a camellia, a type of flower.', 'a photo of a mallow, a type of flower.', 'a photo of a mexican petunia, a type of flower.', 'a photo of a bromelia, a type of flower.', 'a photo of a blanket flower, a type of flower.', 'a photo of a trumpet creeper, a type of flower.', 'a photo of a blackberry lily, a type of flower.']
    if dataset_name == 'cifar100':
        inference_dataset_url = "Multimodal-Fatima/CIFAR100_test"
        training_dataset_url = "my_datasets/cifar100_" + str(shot) + "shot.pt"
        place_holder = 'image'
        unet_input_prompt_size = 15
        if_l1 = True
        prompts = ['a blurry photo of a apple.', 'a blurry photo of a aquarium fish.', 'a blurry photo of a baby.', 'a blurry photo of a bear.', 'a blurry photo of a beaver.', 'a blurry photo of a bed.', 'a blurry photo of a bee.', 'a blurry photo of a beetle.', 'a blurry photo of a bicycle.', 'a blurry photo of a bottle.', 'a blurry photo of a bowl.', 'a blurry photo of a boy.', 'a blurry photo of a bridge.', 'a blurry photo of a bus.', 'a blurry photo of a butterfly.', 'a blurry photo of a camel.', 'a blurry photo of a can.', 'a blurry photo of a castle.', 'a blurry photo of a caterpillar.', 'a blurry photo of a cattle.', 'a blurry photo of a chair.', 'a blurry photo of a chimpanzee.', 'a blurry photo of a clock.', 'a blurry photo of a cloud.', 'a blurry photo of a cockroach.', 'a blurry photo of a couch.', 'a blurry photo of a crab.', 'a blurry photo of a crocodile.', 'a blurry photo of a cup.', 'a blurry photo of a dinosaur.', 'a blurry photo of a dolphin.', 'a blurry photo of a elephant.', 'a blurry photo of a flatfish.', 'a blurry photo of a forest.', 'a blurry photo of a fox.', 'a blurry photo of a girl.', 'a blurry photo of a hamster.', 'a blurry photo of a house.', 'a blurry photo of a kangaroo.', 'a blurry photo of a keyboard.', 'a blurry photo of a lamp.', 'a blurry photo of a lawn mower.', 'a blurry photo of a leopard.', 'a blurry photo of a lion.', 'a blurry photo of a lizard.', 'a blurry photo of a lobster.', 'a blurry photo of a man.', 'a blurry photo of a maple tree.', 'a blurry photo of a motorcycle.', 'a blurry photo of a mountain.', 'a blurry photo of a mouse.', 'a blurry photo of a mushroom.', 'a blurry photo of a oak tree.', 'a blurry photo of a orange.', 'a blurry photo of a orchid.', 'a blurry photo of a otter.', 'a blurry photo of a palm tree.', 'a blurry photo of a pear.', 'a blurry photo of a pickup truck.', 'a blurry photo of a pine tree.', 'a blurry photo of a plain.', 'a blurry photo of a plate.', 'a blurry photo of a poppy.', 'a blurry photo of a porcupine.', 'a blurry photo of a possum.', 'a blurry photo of a rabbit.', 'a blurry photo of a raccoon.', 'a blurry photo of a ray.', 'a blurry photo of a road.', 'a blurry photo of a rocket.', 'a blurry photo of a rose.', 'a blurry photo of a sea.', 'a blurry photo of a seal.', 'a blurry photo of a shark.', 'a blurry photo of a shrew.', 'a blurry photo of a skunk.', 'a blurry photo of a skyscraper.', 'a blurry photo of a snail.', 'a blurry photo of a snake.', 'a blurry photo of a spider.', 'a blurry photo of a squirrel.', 'a blurry photo of a streetcar.', 'a blurry photo of a sunflower.', 'a blurry photo of a sweet pepper.', 'a blurry photo of a table.', 'a blurry photo of a tank.', 'a blurry photo of a telephone.', 'a blurry photo of a television.', 'a blurry photo of a tiger.', 'a blurry photo of a tractor.', 'a blurry photo of a train.', 'a blurry photo of a trout.', 'a blurry photo of a tulip.', 'a blurry photo of a turtle.', 'a blurry photo of a wardrobe.', 'a blurry photo of a whale.', 'a blurry photo of a willow tree.', 'a blurry photo of a wolf.', 'a blurry photo of a woman.', 'a blurry photo of a worm.']
    
    
    fewshot_data = torch.load(training_dataset_url, map_location="cpu", weights_only=True)
    training_set = FewShotDatasetWrapper(fewshot_data)
    ds = load_dataset(inference_dataset_url, split='test')
    inference_dataset = InferenceDataset(ds, place_holder)

    train_sampler = DistributedSampler(training_set)
    test_sampler = NonUniformDistributedSampler(
        inference_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )
    
    train_loader = DataLoader(
        training_set,
        batch_size=1,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        inference_dataset,
        batch_size=1,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=False
    )

    ddp_noises = DDP(Noises().to(device), device_ids=[device])
    ddp_unet = DDP(UNet().to(device), device_ids=[device])
    optimizer = optim.Adam([{'params': ddp_noises.parameters(), 'lr': 1e-2}, {'params': ddp_unet.parameters(), 'lr': 1e-3}])


    correct, total = inference(diffusion_model, test_loader, device, args.seed, rank, prompts, prompt_bz, if_l1, ddp_noises, ddp_unet, t_values, if_image_specific=True)
    if rank == 0:
        print(f"Before Training Test Acc: {correct / total:.4f} | {correct}/{total}\n")


    for epoch in range(epochs):
    #     if rank == 0:
    #         with torch.no_grad():
    #             current_noise = ddp_noises(t_values[0])
    #             # print(current_noise.mean().item(), current_noise.var().item(), current_noise.norm().item())
    #             print(ratio_high_freq(current_noise))
        train_sampler.set_epoch(epoch)
        loss = train_epoch(diffusion_model, train_loader, optimizer, device, ddp_noises, args.seed, rank, epoch, prompts, unet_input_prompt_size, if_l1, ddp_unet, t_values, if_image_specific=True)
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss:.4f}\n")

        if (epoch + 1) % args.infer_freq == 0:
            correct, total = inference(diffusion_model, test_loader, device, args.seed, rank, prompts, prompt_bz, if_l1, ddp_noises, ddp_unet, t_values, if_image_specific=True)
            if rank == 0:
                print(f"Test Acc: {correct / total:.4f} | {correct}/{total}\n")


    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
