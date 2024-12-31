import json
import os
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import shutil
import torch
import networks
from tqdm import tqdm
from utils_scripts import *
from loss_terms import *
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR
from elasticdeform import deform_random_grid
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

# Load parameters
with open("/home/ftagalak/fvdl_ow/Scripts/parameters.json", "r") as f:
    params = json.load(f)

# Hyperparameter Search Space
search_space = {
    "init_lr": tune.grid_search([0.01, 0.001, 0.0001]),
    "batch_size": tune.grid_search([16, 32, 64]),
    "optimizer": tune.grid_search(["AdamW", "SGD"]),
    "scheduler": tune.grid_search(["ReduceLROnPlateau", "CosineAnnealingLR"]),
    "weight_decay": tune.grid_search([1e-5, 1e-6])
}

# Train function for Ray Tune
def train_model(config, checkpoint_dir=None):
    # Load parameters
    BATCH_SIZE = config["batch_size"]
    INIT_LR = config["init_lr"]
    optimizer_type = config["optimizer"]
    scheduler_type = config["scheduler"]
    weight_decay = config["weight_decay"]
    EPOCHS = 100#params['epochs']
    NUM_WORKERS = params['num_workers']
    NETWORK = params['network']
    LOSS = params['loss']
    CURRENT_FOLD = f"fold_{params['fold']}"
    
    # Prepare Dataset
    train_transform = transforms.Compose([
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.12, hue=0.12),
        transforms.ToTensor(),
    ])

    inference_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    train_dataset = DynamicSamplingDataset2(
        csv_file=params['train_csv_file'].format(current_fold=f"fold_{params['fold']}"),
        ssim_matrix=0,
        transformA=train_transform,
        transformB=train_transform
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE
    )

    base_dataset = AllImagesDataset(
        csv_file=[params['test_csv_file'].format(current_fold=CURRENT_FOLD)],
        transform=inference_transform
    )
    base_dataloader = DataLoader(
        base_dataset, shuffle=False, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE
    )

    query_dataset = AllImagesDataset(
        csv_file=[params['test_csv_file'].format(current_fold=CURRENT_FOLD)],
        transform=inference_transform
    )
    query_dataloader = DataLoader(
        query_dataset, shuffle=False, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE
    )

    # Model
    model = networks.SimSiamWrapper(networks.resnet18(pretrained=True, num_classes=492, loss=LOSS, cat_flag=True))
    model.to('cuda')

    # Loss
    criterion = nn.CosineSimilarity(dim=1).cuda()

    # Optimizer
    if optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9, nesterov=True)

    # Scheduler
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    kll = SymmetricKLDivLoss()

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (img1, img2, labels, ids) in enumerate(tqdm(train_dataloader)):
            img1, img2 = img1.cuda(), img2.cuda()
            optimizer.zero_grad()

            p1, p2, z1, z2, y1, y2 = model(img1, img2)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            loss += 2.5 * kll(p1, p2)
            loss += 0.25 * orthogonality_loss(model.simple_siamese.predictor[-1].weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)
        scheduler.step(epoch_loss)

        if epoch>=10:

            val_loss = faiss_inf_ow(model, base_dataloader=base_dataloader, query_dataloader=query_dataloader)
        
            # Report metrics to Ray Tune
            session.report({"loss": val_loss})

# Scheduler for Early Stopping
asha_scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=params['epochs'],
    grace_period=40,
    reduction_factor=2
)

# Run the Grid Search
tune.run(
    train_model,
    config=search_space,
    resources_per_trial={"cpu": 8, "gpu": 2},  # Use 3 GPUs
    scheduler=asha_scheduler,
    num_samples=1,  # Number of trials for each configuration
    local_dir="/home/ftagalak/fvdl_ow/RayTuneLogs",
    name="GridSearchExperiment2"
)
