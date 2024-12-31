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

### BEST SO FAR: /home/ftagalak/fvdl_ow/Experiments/NewTrainDataset/resnet18_SymmetricKLDivLoss_ORTH_COSSIM_0001_ep250_fold_1_b16_WInone_OPTAdamW
### _2comma5SymmetricKLDivLossp1_ORTH_COSSIM_ -> 0.0121
### /home/ftagalak/fvdl_ow/Experiments/NewTrainDataset/resnet18_2comma50comma5SymmetricKLDivLossp1_ORTH_COSSIM_0001_ep250_fold_1_b16_WInone_OPTAdamW/ -> 0.0114
### /home/ftagalak/fvdl_ow/Experiments/NewTrainDataset/resnet18_2comma50comma25SymmetricKLDivLossp1_ORTH_COSSIM_0001_ep250_fold_1_b16_WInone_OPTAdamW/ -> 0.0095

### resnet18_ORIG075new4IAFL_0001_ep250_fold_1_b16_WInone_OPTAdamW -> 0.0090
### resnet18_ORIG075new6IAFL_0001_ep250_fold_1_b16_WInone_OPTAdamW -> 0.0087
# Load parameters from a JSON file
with open("/home/ftagalak/fvdl_ow/Scripts/parameters.json", "r") as f:
    params = json.load(f)

# Extract parameters
CURRENT_FOLD = f"fold_{params['fold']}"
EPOCHS = int(params['epochs'])
BATCH_SIZE = int(params['batch_size'])
NUM_WORKERS = params['num_workers']
INIT_LR = params['init_lr']
NETWORK = params['network']
LOSS = params['loss']
OUTPUT_FOLDER = os.path.join(
    params['output_folder'],
    f"{NETWORK}_ORIG075new11IAFL_{str(INIT_LR).replace('.', '')}_ep{EPOCHS}_{CURRENT_FOLD}_b{BATCH_SIZE}_WI{params['weights_init']}_OPT{params['optimizer']['type']}/"
)

print(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Copy script files to the output folder
for script in params['scripts_to_copy']:
    shutil.copy(script, OUTPUT_FOLDER)

def initialize_cnn_weights(layer, method='kaiming'):
    """
    Initialize weights for a CNN layer in PyTorch with various methods.

    Args:
    - layer (nn.Module): PyTorch layer to initialize.
    - method (str): Method to use for initialization. Options include 'kaiming', 'xavier', 'uniform',
                    'normal', 'orthogonal', 'sparse', 'constant', 'zeros', 'ones'.

    Returns:
    - None: Modifies the layer in place.
    """
    if isinstance(layer, nn.Conv2d):
        if method == 'kaiming':
            torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        elif method == 'xavier':
            torch.nn.init.xavier_uniform_(layer.weight)
        elif method == 'uniform':
            torch.nn.init.uniform_(layer.weight)
        elif method == 'normal':
            torch.nn.init.normal_(layer.weight)
        elif method == 'orthogonal':
            torch.nn.init.orthogonal_(layer.weight)
        elif method == 'he':
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu') 
        else:
            raise ValueError("Unknown initialization method")

        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

# Network selection
if NETWORK == 'resnet18':
    encoder = networks.resnet18(pretrained=True, num_classes=492, loss=LOSS, cat_flag=True)
elif NETWORK == 'resnet34':
    encoder = networks.resnet34(pretrained=True, num_classes=492, loss=LOSS)

model = networks.SimSiamWrapper(encoder)
if params['weights_init']!='none':
    print('!!! CNN weights get initialized !!!')
    model.apply(lambda layer: initialize_cnn_weights(layer, method='normal'))
model.to('cuda')

# Create a SummaryWriter instance
writer = SummaryWriter(log_dir=OUTPUT_FOLDER)

elastic_transform = transforms.Lambda(lambda x: 
    torch.tensor(deform_random_grid(x.numpy(), sigma=3, points=5), dtype=torch.float32)
)

def high_pass_filter(img, cutoff=0.1):
    fft = torch.fft.fft2(img)  # Compute FFT
    fft_shifted = torch.fft.fftshift(fft)  # Shift zero frequency to center

    # Create filter mask
    h, w = img.shape[-2:]
    mask = torch.ones_like(fft_shifted)
    cy, cx = h // 2, w // 2  # Center coordinates
    radius = int(cutoff * min(h, w))  # Cutoff frequency
    mask[cy-radius:cy+radius, cx-radius:cx+radius] = 0  # Suppress low frequencies

    # Apply mask
    filtered = fft_shifted * mask
    filtered = torch.fft.ifftshift(filtered)  # Reverse shift
    img_filtered = torch.fft.ifft2(filtered).real  # Inverse FFT

    return img_filtered

def random_phase_distortion(img, scale=0.2):
    fft = torch.fft.fft2(img)
    phase = torch.angle(fft)
    magnitude = torch.abs(fft)

    # Add random phase noise
    random_phase = torch.rand_like(phase) * 2 * torch.pi * scale
    phase_shifted = phase + random_phase

    # Reconstruct image
    distorted = magnitude * torch.exp(1j * phase_shifted)
    img_distorted = torch.fft.ifft2(distorted).real
    return img_distorted

# Define transformations
train_transform = transforms.Compose([
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.RandomRotation(3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.12, hue=0.12),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: high_pass_filter(x, cutoff=0.1)),
    transforms.Lambda(lambda x: random_phase_distortion(x, scale=0.02)),
])

inference_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Dataset and DataLoader
train_dataset = DynamicSamplingDataset2(
    csv_file=params['train_csv_file'].format(current_fold=CURRENT_FOLD),
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

# Loss
criterion = nn.CosineSimilarity(dim=1).cuda()
loss_terms = {
    "CosineSimilarity": nn.CosineSimilarity(dim=1).cuda(),
    "SymmetricKLDivLoss": SymmetricKLDivLoss(),
    "CrossCovarianceLoss": CrossCovarianceLoss(),
    "CosFaceLoss": CosFace(),
    "PrototypeLearning": PrototypeLearning(num_classes=492,
                                           embedding_dim=512),
    "DirectionalContrastiveLoss": DirectionalContrastiveLoss(),
    "ArccosineCenterLoss": ArccosineCenterLoss(492//2, 512),
    "AngularMarginContrastiveLoss": AngularMarginContrastiveLoss(),
    "FFT_Loss": FrequencyLoss(),
    "AdaptiveWeightedLoss":AdaptiveWeightedLoss()
}
loss_weights = params["loss_terms"]


# Optimizer
optimizer_params = params["optimizer"]
if optimizer_params["type"] == "AdamW":
    optimizer = optim.AdamW(list(model.parameters()), lr=INIT_LR)#, weight_decay=optimizer_params["weight_decay"])
elif optimizer_params["type"] == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=optimizer_params["options"]["SGD"]["momentum"], nesterov=optimizer_params["options"]["SGD"]["nesterov"])
elif optimizer_params["type"] == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr=INIT_LR, momentum=optimizer_params["options"]["RMSprop"]["momentum"], alpha=optimizer_params["options"]["RMSprop"]["alpha"])
elif optimizer_params["type"] == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR, betas=optimizer_params["options"]["Adam"]["betas"])
else:
    raise ValueError(f"Unsupported optimizer: {optimizer_params['type']}")

# Scheduler
scheduler_params = params["scheduler"]
if scheduler_params["type"] == "ReduceLROnPlateau":
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=scheduler_params["options"]["ReduceLROnPlateau"]["mode"],
        factor=scheduler_params["options"]["ReduceLROnPlateau"]["factor"],
        patience=scheduler_params["options"]["ReduceLROnPlateau"]["patience"],
        verbose=scheduler_params["options"]["ReduceLROnPlateau"]["verbose"],
        min_lr=scheduler_params["options"]["ReduceLROnPlateau"]["min_lr"]
    )
elif scheduler_params["type"] == "CosineAnnealingLR":
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_params["T_max"], eta_min=scheduler_params["eta_min"])
elif scheduler_params["type"] == "StepLR":
    scheduler = StepLR(optimizer, step_size=scheduler_params["options"]["StepLR"]["step_size"], gamma=scheduler_params["options"]["StepLR"]["gamma"])
elif scheduler_params["type"] == "OneCycleLR":
    scheduler = OneCycleLR(optimizer, max_lr=scheduler_params["options"]["OneCycleLR"]["max_lr"], steps_per_epoch=scheduler_params["options"]["OneCycleLR"]["steps_per_epoch"], epochs=scheduler_params["options"]["OneCycleLR"]["epochs"])
else:
    raise ValueError(f"Unsupported scheduler: {scheduler_params['type']}")

smallest_train_loss = np.inf
smallest_val_loss = np.inf
train_losses = []
val_losses = []

# Train the Siamese network
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_idx, (img1, img2, labels, ids) in enumerate(tqdm(train_dataloader)):
        img1, img2, labels, ids = img1.cuda(), img2.cuda(), labels.cuda(), ids.cuda()
        optimizer.zero_grad()
        p1, p2, z1, z2, y1, y2 = model(x1 = img1, x2 = img2)
        # p1, p2, z1, z2 = model(x1 = img1, x2 = img2)
        
        loss = 0
        for term, weight in loss_weights.items():
            if term == "CosineSimilarity" and weight["weight"]!=0:
                loss += weight["weight"] * (-(loss_terms[term](p1, z2).mean() + loss_terms[term](p2, z1).mean()) * 0.5)
            elif term == "SymmetricKLDivLoss" and weight["weight"]!=0:
                loss += weight["weight"] * loss_terms[term](p1, p2)
            elif term == "CrossCovarianceLoss" and weight["weight"]!=0:
                loss += weight["weight"] * loss_terms[term](z1, z2)
            elif term == "InverseActivationFrequencyLoss" and weight["weight"]!=0:
                loss += weight["weight"] * (inverse_activation_frequency_loss(p1) + inverse_activation_frequency_loss(p2))/ 2
            elif term == "WeightsOrthogonalityLoss" and weight["weight"]!=0:
                loss += weight["weight"] * orthogonality_loss(model.simple_siamese.predictor[-1].weight)
            elif term == "CosFaceLoss" and weight["weight"]!=0:
                loss += weight["weight"] * (loss_terms[term](p1, ids) + loss_terms[term](p2, ids))/2
            elif term == "AVR" and weight["weight"]!=0:
                loss += weight["weight"] * (compute_angular_variance(p1, ids) + compute_angular_variance(p2, ids))/2
            elif term == "PrototypeLearning" and weight["weight"]!=0:
                loss += weight["weight"] * (loss_terms[term](p1, ids) + loss_terms[term](p2, ids))/2
            elif term == "DirectionalContrastiveLoss" and weight["weight"]!=0:
                loss += weight["weight"] * loss_terms[term](p1, p2, ids)
            elif term == "ArccosineCenterLoss" and weight["weight"]!=0:
                loss += weight["weight"] * (loss_terms[term](p1,ids) + loss_terms[term](p2,ids))/2
            elif term == "AngularMarginContrastiveLoss" and weight["weight"]!=0:
                loss += weight["weight"] * (loss_terms[term](p1, ids) + loss_terms[term](p2, ids))/2
            elif term == "FFT_Loss" and weight["weight"]!=0:
                loss += weight["weight"] * loss_terms[term](y1, y2)
            elif term == "AdaptiveWeightedLoss" and weight["weight"]!=0:
                loss += weight["weight"] * loss_terms[term](p1, p2)

        loss.backward()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate', current_lr, epoch)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    if train_loss<=smallest_train_loss:
        smallest_train_loss = train_loss
    writer.add_scalar('Loss/train', train_loss, epoch)
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss:.4f}")

    if (smallest_train_loss == train_loss and epoch>=150) or epoch>=10:

        val_loss = faiss_inf_ow(model, base_dataloader=base_dataloader, query_dataloader=query_dataloader)
        
        print('Val Loss: {:.4f}'.format(val_loss))
        val_losses.append(val_loss)
        writer.add_scalar('Loss/val', val_loss, epoch)

        # Save the model if the evaluation score is better
        if val_loss < smallest_val_loss:
            smallest_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER, 'best_val_loss_model_ep_' +str(epoch).zfill(3) + '.pt'))

    scheduler.step(train_loss)
