import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import ClassConditionedUnet
import logging
import argparse
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from evaluator import evaluation_model
from utils import iclevr_dataset
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

accelerator = Accelerator()


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, object_OH, noise_scheduler):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

                # Get model pred
                with torch.no_grad():
                    residual = model(x, t, object_OH)  # Again, note that we pass in our labels y

                # Update sample with step
                x = noise_scheduler.step(residual, t, x).prev_sample

                
        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    
    train_data = iclevr_dataset(args)
    train_dataloader = DataLoader(train_data,args.batch,shuffle=True)
    print(f"training data len : {len(train_dataloader)}")

    device = args.device
    model = ClassConditionedUnet(num_classes = 24).to(device)

    model.load_state_dict(torch.load(os.path.join("models", args.load_name, f"ckpt.pt")))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("logs", args.run_name))
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    l = len(train_dataloader)
    min_mse = 10

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        mse_list = []
        for i, (images, object_OH) in enumerate(pbar):
            images = images.to(device) # GT image
            object_OH = object_OH.float().to(device) # object one hot

            noise = torch.randn_like(images)
            timesteps = torch.randint(0, 999, (images.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(images, noise, timesteps)

            predicted_noise = model(noisy_x, timesteps, object_OH)
            
            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            mse_list.append(loss.item())
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        mean_mse = sum(mse_list[-100:])/100
        logger.add_scalar("MSE", mean_mse, epoch)
        sampled_images = diffusion.sample(model, images.shape[0], object_OH,noise_scheduler)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

def test(args):
    test_data = iclevr_dataset(args)
    test_dataloader = DataLoader(test_data,args.test_batch,shuffle=False)
    device = args.device
    model = ClassConditionedUnet(num_classes = 24).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

    model.load_state_dict(torch.load(os.path.join("models", args.load_name, f"ckpt.pt")))
    diffusion = Diffusion(img_size=args.image_size, device=device)

    model.eval()
    l = len(test_dataloader)
    pbar = tqdm(test_dataloader)

    E = evaluation_model()
    acc_list = []
    x_list = []
    with torch.no_grad():
        for i, ( _, object_OH) in enumerate(pbar):
            object_OH = object_OH.float().to(device) # object one hot
            x = diffusion.sample(model, args.test_batch, object_OH,noise_scheduler)
            # x tensor [0,1] float
            save_images(x,os.path.join("test", args.run_name, f"{args.mode}_{i}.jpg"))

            acc = E.eval(x,object_OH)
            acc_list.append(acc)
            x_list.append((x.clamp(-1, 1) + 1) / 2)
        
    x = x_list[0]
    for i in range(1,len(x_list)):
        x = torch.cat((x, x_list[i]), 0)
    
    save_images(x,os.path.join("test", args.run_name, f"{args.mode}_{i}.jpg"))
    print(f"accuracy:{float(sum(acc_list))/l}")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",type = str, default = "Diffuser_embedding")
    parser.add_argument("--load_name",type = str, default = "Diffuser_Yang")
    parser.add_argument("--epochs",type = int, default = 500)
    parser.add_argument("--batch",type = int, default = 24)
    parser.add_argument("--test_batch",type = int, default = 32)
    parser.add_argument("--image_size",type = int, default = 64)
    parser.add_argument("--lr",type = float, default = 0.005)
    parser.add_argument("--device",type = str, default = "cuda")
    parser.add_argument("--data_root",type = str, default = "./iclevr")
    parser.add_argument("--mode",type = str, default = "train") # train , test , new_test


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parser()
    args.device = accelerator.device

    if args.mode == "train":
        train(args)
    else:
        args.run_name = args.load_name
        test(args)
