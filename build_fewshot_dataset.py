import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from datasets import load_dataset, concatenate_datasets
from PIL import Image
from torchvision import transforms as tvt
import random

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def load_image(pil_img, target_size=None):
    """
    """
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # (1, C, H, W)

@torch.no_grad()
def img_to_latents(x, vae):
    """
    """
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents

def build_fewshot_dataset(dataset_url, shot, img_placeholder, diffusion_model, seed=42):
    set_seed(seed)
    
    dataset = load_dataset(dataset_url, split='train')
    

    pipeline = StableDiffusionPipeline.from_pretrained(diffusion_model, torch_dtype=torch.float32)
    pipeline.to("cuda:0")
    vae = pipeline.vae
    
    data = []
    if 'Pets' in dataset_url:
        cate_num = 37
    elif 'cifar10' in dataset_url:
        cate_num = 10
    elif 'CIFAR100' in dataset_url:
        cate_num = 100
    elif 'Aircraft' in dataset_url:
        cate_num = 100
    elif 'Caltech' in dataset_url:
        cate_num = 102
    elif 'DTD' in dataset_url:
        cate_num = 47
    elif 'stl' in dataset_url:
        cate_num = 10
    elif 'eurosat' in dataset_url:
        cate_num = 10
    elif 'flowers' in dataset_url:
        cate_num = 102
    
    if shot==16 and 'flowers' in dataset_url:
        dataset2 = load_dataset(dataset_url, split='validation')
        dataset = concatenate_datasets([dataset, dataset2])


    label_indices = {i: [] for i in range(cate_num)}
    for idx in range(len(dataset)):
        label = dataset[idx]['label']
        label_indices[label].append(idx)

    for label, indices in label_indices.items():
        selected_indices = random.sample(indices, shot)
        for idx in selected_indices:
            sample = dataset[idx]
            image = sample[img_placeholder].convert('RGB')
            input_img = load_image(image, 512).to("cuda:0")
            x0 = img_to_latents(input_img, vae)
            x0 = x0.cpu()
            data.append((x0, label))
    return data

if __name__ == '__main__':
    dataset_url = "Multimodal-Fatima/CIFAR100_train"
    shot = 16
    diffusion_model = "stabilityai/stable-diffusion-2-base"
    img_placeholder = "image"
    
    fewshot_data = build_fewshot_dataset(dataset_url, shot, img_placeholder, diffusion_model, seed=42)
    torch.save(fewshot_data, "./my_datasets/cifar100_"+ str(shot) +"shot.pt")
    print("Few-shot dataset saved to fewshot_dataset.pt")
