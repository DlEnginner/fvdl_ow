import torch
import torch.nn.functional as F
from itertools import combinations
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import math
import random
import itertools
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import sklearn.metrics
import numpy as np
from skimage.filters import gabor
from skimage.util import img_as_ubyte
from PIL import Image
from skimage.filters import frangi, hessian
from scipy.signal import wiener
from skimage.restoration import denoise_bilateral
from skimage.filters import unsharp_mask
from skimage.morphology import opening, closing, disk
from skimage.exposure import equalize_adapthist
from itertools import product
from sklearn.metrics import accuracy_score, recall_score
from tqdm import tqdm
from random import random
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
# from sklearn.metrics import presicion_score
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import faiss
from sklearn import preprocessing, metrics

RANDOM_STATE = 123

def prototype_orthogonality_loss(embeddings, labels):
    """
    Promotes orthogonality of embeddings with the prototypes of other classes.

    Args:
    - embeddings (torch.Tensor): Tensor of shape (batch_size, embedding_dim).
    - labels (torch.Tensor): Tensor of shape (batch_size,) with class IDs.

    Returns:
    - torch.Tensor: Scalar loss value.
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Get unique classes in the batch
    unique_classes = labels.unique()

    # Compute prototypes for each class
    prototypes = []
    for cls in unique_classes:
        class_mask = labels == cls  # Mask for class `cls`
        class_embeddings = embeddings[class_mask]  # Embeddings belonging to class `cls`
        prototype = class_embeddings.mean(dim=0)  # Mean embedding (prototype)
        prototypes.append(F.normalize(prototype, p=2, dim=0))  # Normalize prototype

    # Stack prototypes for efficient computation
    prototypes = torch.stack(prototypes)  # Shape: (num_classes_in_batch, embedding_dim)

    # Compute similarity between embeddings and prototypes
    similarity_matrix = torch.matmul(embeddings, prototypes.T)  # Shape: (batch_size, num_classes_in_batch)

    # Create mask to exclude similarity with own class prototype
    prototype_class_map = unique_classes.repeat(similarity_matrix.size(0), 1)  # Shape: (batch_size, num_classes_in_batch)
    label_matrix = labels.unsqueeze(1).expand_as(prototype_class_map)  # Shape: (batch_size, num_classes_in_batch)
    mask = prototype_class_map != label_matrix  # Mask for non-class prototypes

    # Penalize non-orthogonality to other class prototypes
    loss = torch.sum((similarity_matrix[mask]) ** 2) / mask.sum()

    return loss


def inter_class_separation_loss(embeddings, ids):
    """
    Encourages embeddings of different classes to be dissimilar.
    
    Args:
        embeddings (torch.Tensor): Batch of embeddings (N, D), where N is batch size and D is embedding dimension.
        ids (torch.Tensor): Class IDs corresponding to each embedding (N,).
    
    Returns:
        torch.Tensor: Separation loss scalar.
    """
    # Normalize embeddings for cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute cosine similarity matrix (N x N)
    sim_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Mask to identify negative pairs (N x N)
    neg_mask = ids.unsqueeze(1) != ids.unsqueeze(0)  # 1 where negative pairs, 0 otherwise
    
    # Apply mask to keep only negative similarities
    neg_sim = sim_matrix * neg_mask  # Retain only negative similarities
    neg_sim_exp = torch.exp(neg_sim)  # Exponentiate for loss
    
    # Average over negatives for each embedding
    num_negatives = neg_mask.sum(dim=1).clamp_min(1.0)  # Avoid division by zero
    loss = torch.sum(neg_sim_exp, dim=1) / num_negatives  # Sum and normalize by negatives
    loss = loss.mean()  # Average over all embeddings
    
    return loss

def embedding_uniformity_loss(embeddings, labels, temperature=0.001):
    # Normalize embeddings to lie on the unit hypersphere
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Pairwise squared Euclidean distances
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2) ** 2  # Shape: [N, N]

    # Create mask to exclude same-class pairs
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # Shape: [N, N]
    mask = ~label_matrix  # Exclude same-class pairs (invert the mask)

    # Apply mask to distances
    masked_distances = pairwise_distances * mask.float()

    # Exponential similarity scaled by temperature
    exp_sim = torch.exp(-masked_distances / temperature)

    # Exclude self-similarity by creating a separate mask (no in-place modification)
    identity_mask = torch.eye(exp_sim.size(0), device=exp_sim.device).bool()
    exp_sim = exp_sim.masked_fill(identity_mask, 0)

    # Compute uniformity loss
    return torch.log(torch.sum(exp_sim))

def embedding_uniformity_loss_mean(embeddings, labels, temperature=0.1):
    # Normalize embeddings to lie on the unit hypersphere
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Pairwise squared Euclidean distances
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2) ** 2  # Shape: [N, N]

    # Create mask to exclude same-class pairs
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # Shape: [N, N]
    mask = ~label_matrix  # Exclude same-class pairs (invert the mask)

    # Apply mask to distances
    masked_distances = pairwise_distances * mask.float()

    # Exponential similarity scaled by temperature
    exp_sim = torch.exp(-masked_distances / temperature)

    # Exclude self-similarity
    identity_mask = torch.eye(exp_sim.size(0), device=exp_sim.device).bool()
    exp_sim = exp_sim.masked_fill(identity_mask, 0)

    # Compute mean exponential similarity
    return torch.mean(exp_sim)

def embedding_variance_uniformity_loss_regularized(embeddings, labels):
    # Normalize embeddings to lie on the unit hypersphere
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Pairwise distances
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)  # Shape: [N, N]

    # Mask to exclude same-class pairs
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # Shape: [N, N]
    mask = ~label_matrix  # Exclude same-class pairs (invert the mask)

    # Extract distances for different-class pairs
    different_class_distances = pairwise_distances[mask]

    # Handle empty distances
    if different_class_distances.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)  # No different-class pairs

    # Regularize distances to prevent instability
    different_class_distances = torch.clamp(different_class_distances, min=1e-6)

    # Compute variance of distances
    uniformity_loss = torch.var(different_class_distances)
    return uniformity_loss


def embedding_inverse_distance_uniformity_loss(embeddings, labels):
    # Normalize embeddings to lie on the unit hypersphere
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Pairwise squared Euclidean distances
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2) ** 2  # Shape: [N, N]

    # Mask to exclude same-class pairs
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # Shape: [N, N]
    mask = ~label_matrix  # Exclude same-class pairs (invert the mask)

    # Check if mask is empty (no valid different-class pairs)
    if mask.float().sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    # Compute inverse distances with numerical stability
    inverse_distances = 1 / (1 + pairwise_distances + 1e-6)

    # Compute uniformity loss
    uniformity_loss = torch.sum(inverse_distances * mask.float()) / mask.float().sum()
    return uniformity_loss

def embedding_inverse_distance_uniformity_loss_improved(embeddings, labels, temperature=1.0):
    """
    Computes an inverse-distance-based uniformity loss.
    Embeddings from the same class are excluded from penalties.
    """
    # Normalize embeddings to lie on the unit hypersphere
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Pairwise Euclidean distances
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)  # Shape: [N, N]

    # Create mask to exclude same-class pairs
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # Shape: [N, N]
    mask = ~label_matrix  # Exclude same-class pairs (invert the mask)

    # Check if there are any valid different-class pairs
    if mask.float().sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    # Compute soft-inverse penalty for distances
    inverse_penalty = temperature / (1 + pairwise_distances + 1e-6)  # Add epsilon for stability

    # Apply the mask
    masked_penalty = inverse_penalty * mask.float()

    # Compute loss: average penalty over all valid different-class pairs
    uniformity_loss = torch.sum(masked_penalty) / mask.float().sum()

    return uniformity_loss


def compute_laplacian_smoothness_loss(features):
    # Create the Laplacian kernel
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32, device=features.device).unsqueeze(0).unsqueeze(0)

    # Expand the kernel to match the number of channels
    laplacian_kernel = laplacian_kernel.repeat(features.shape[1], 1, 1, 1)  # [C, 1, 3, 3]

    # Apply Laplacian convolution across all channels
    laplacian = nn.functional.conv2d(features, laplacian_kernel, padding=1, groups=features.shape[1])

    # Compute the smoothness loss as the mean squared Laplacian
    smoothness_loss = torch.mean(laplacian ** 2)
    return smoothness_loss

class ArccosineCenterLoss(nn.Module):
    def __init__(self, num_classes=492, feat_dim=512):
        super(ArccosineCenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)).to('cuda')

    def forward(self, features, labels):
        # Normalize the feature vectors and the centers
        normalized_features = F.normalize(features, p=2, dim=1)
        normalized_centers = F.normalize(self.centers, p=2, dim=1)

        # Compute cosine similarities between features and corresponding centers
        cos_theta = F.linear(normalized_features, normalized_centers)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Clamping to avoid numerical issues

        # Gather only the cosines of the correct centers
        target_labels = labels.unsqueeze(1).expand_as(cos_theta)
        cos_theta_target = cos_theta.gather(1, target_labels).squeeze(1)

        # Compute the arccosine (angular distance)
        theta = torch.acos(cos_theta_target)

        # Average the angular distances
        arccosine_loss = torch.mean(theta)

        return arccosine_loss

def calculate_cross_entropy_loss_with_one_hot(embeddings, labels):
    """
    Calculate the cross-entropy loss given embeddings and labels.

    Args:
    embeddings (torch.Tensor): The embeddings (logits) of the model. Shape: [batch_size, num_classes].
    labels (torch.Tensor): The true labels. Shape: [batch_size]

    Returns:
    torch.Tensor: The calculated cross-entropy loss.
    """
    # Apply softmax to embeddings to get probabilities
    probs = F.softmax(embeddings, dim=1)

    # Convert labels to one-hot encoding
    num_classes = embeddings.size(1)
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).to(embeddings.dtype)

    # Compute the log of probabilities
    log_probs = torch.log(probs + 1e-6)  # Adding a small value to prevent log(0)

    # Calculate the negative log likelihood loss with the one-hot encoded labels
    loss = -1 * torch.sum(one_hot_labels * log_probs, dim=1)
    loss = torch.mean(loss)

    return loss


'''
class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes=492, embedding_size=128, margin, scale):
        """
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        (https://arxiv.org/pdf/1801.07698.pdf)
        Args:
            num_classes: The number of classes in your training dataset
            embedding_size: The size of the embeddings that you pass into
            margin: m in the paper, the angular margin penalty in radians
            scale: s in the paper, feature scale
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        
        self.W = torch.nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.W)
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (None, embedding_size)
            labels: (None,)
        Returns:
            loss: scalar
        """
        cosine = self.get_cosine(embeddings) # (None, n_classes)
        mask = self.get_target_mask(labels) # (None, n_classes)
        cosine_of_target_classes = cosine[mask == 1] # (None, )
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        ) # (None, )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1) # (None,1)
        logits = cosine + (mask * diff) # (None, n_classes)
        logits = self.scale_logits(logits) # (None, n_classes)
        return nn.CrossEntropyLoss()(logits, labels)
        
    def get_cosine(self, embeddings):
        """
        Args:
            embeddings: (None, embedding_size)
        Returns:
            cosine: (None, n_classes)
        """
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.W))
        return cosine
    
    def get_target_mask(self, labels):
        """
        Args:
            labels: (None,)
        Returns:
            mask: (None, n_classes)
        """
        batch_size = labels.size(0)
        onehot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(-1), 1)
        return onehot
        
    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        """
        Args:
            cosine_of_target_classes: (None,)
        Returns:
            modified_cosine_of_target_classes: (None,)
        """
        eps = 1e-6
        # theta in the paper
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + eps, 1 - eps))
        return torch.cos(angles + self.margin)
    
    def scale_logits(self, logits):
        """
        Args:
            logits: (None, n_classes)
        Returns:
            scaled_logits: (None, n_classes)
        """
        return logits * self.scale
'''

class SoftmaxLoss(nn.Module):
    def __init__(self, num_classes, embedding_size):
        """
        Regular softmax loss (1 fc layer without bias + CrossEntropyLoss)
        Args:
            num_classes: The number of classes in your training dataset
            embedding_size: The size of the embeddings that you pass into
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        
        self.W = torch.nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.W)
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (None, embedding_size)
            labels: (None,)
        Returns:
            loss: scalar
        """
        logits = F.linear(embeddings, self.W)
        return nn.CrossEntropyLoss()(logits, labels)

def compute_eer(label, pred, positive_label=1):
    """
    Python compute equal error rate (eer)
    ONLY tested on binary classification

    :param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
    :param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
    :param positive_label: the class that is viewed as positive class when computing EER
    :return: equal error rate (EER)
    """
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred)#, positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer

def compute_eer2(label, pred, positive_label=1):
    fpr, tpr, _  = roc_curve(label, pred, pos_label = positive_label)
    eer = brentq(lambda x: 1.0 - x -interp1d(fpr,tpr)(x), 0.0, 1.0)
    return eer

def normalized_cross_corr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    x = x - mean_x
    y = y - mean_y
    correlation = torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))
    return correlation

class InferenceTestSet(Dataset):
    def __init__(self, test_csv_file, transform=None, seed=RANDOM_STATE):
        print('InferenceTestSet')
        self.df = pd.read_csv(test_csv_file)
        self.transform = transform

        self.unique_ids = self.df['id'].unique()
        self.num_unique_ids = len(self.unique_ids)

        # Set seeds
        np.random.seed(seed)
        random.seed(seed)

        self._load_pairs()

    def _load_pairs(self):
        self.pairs = []
        self.labels = []
        for idx, id in enumerate(self.df['id'].unique()):
            matched = self.df.loc[self.df['id']==id]
            non_matched = self.df.loc[self.df['id']!=id]
            unique_pairs = list(itertools.combinations(matched.index, 2))
            assert len(unique_pairs)==1, "More than 2 rows in the matched dataframe!!!"
            for pair in unique_pairs:
                self.pairs.append([self.df.iloc[pair[0]]['image_path'], self.df.iloc[pair[1]]['image_path']])
                # print(self.df.iloc[pair[0]]['image_path'], self.df.iloc[pair[1]]['image_path'], 1)
                self.labels.append(1)
                # The seed for the random state for the sample is determined by the current id index
                self.pairs.append([self.df.iloc[pair[random.randint(0,1)]]['image_path'], non_matched.sample(1, random_state=idx)['image_path'].tolist()[0]])
                # print(self.df.iloc[pair[random.randint(0,1)]]['image_path'], non_matched.sample(1, random_state=idx)['image_path'].tolist()[0], 0)
                self.labels.append(0)
                # print()

    def __getitem__(self, index):
        # Open images
        image1 = Image.open(self.pairs[index][1])
        image2 = Image.open(self.pairs[index][0])

        label = self.labels[index]

        # Apply transform if any
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, label

    def __len__(self):
        return len(self.pairs)

class AllImagesDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        if len(csv_file)==0:
            self.database = pd.read_csv(csv_file[0])
        else:
            lista = []
            for i in range(len(csv_file)):
                lista.append(pd.read_csv(csv_file[i]))
            self.database = pd.concat(lista)
        self.transform = transform

        self.image_paths = self.database['image_path'].tolist()
        self.labels = self.database['id'].tolist()

        assert len(self.image_paths) == len(self.labels)
        assert len(self.image_paths)%2==0

        self.pairs = []
        self.pair_labels = []

        # Create random pairs
        while len(self.image_paths) > 1:
            img1 = self.image_paths.pop()
            label1 = self.labels.pop()
            img2 = self.image_paths.pop()
            label2 = self.labels.pop()
            self.pairs.append((img1, img2))
            self.pair_labels.append((label1, label2))
        
        assert len(self.pairs) == len(self.pair_labels), "Different number of labels and image pairs"

    def __getitem__(self, index):
        # Open images
        image1 = Image.open(self.pairs[index][0])
        image2 = Image.open(self.pairs[index][1])

        label1, label2 = self.pair_labels[index]

        # Apply transform if any
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.pairs)


class InferenceDataset(Dataset):
    def __init__(self, database_csv_file, query_csv_file, transform=None):
        self.database = pd.read_csv(database_csv_file)
        self.query = pd.read_csv(query_csv_file)
        self.transform = transform

        self.unique_ids = self.database['id'].unique()
        self.num_unique_ids = len(self.unique_ids)

        self._load_pairs()

    def _load_pairs(self):
        self.pairs = []
        self.labels = []
        for i in range(len(self.query)):
            id = self.query.iloc[i]['id']
            # Keep all matches:
            matched = self.database.loc[self.database['id']==id]
            for j in range(len(matched)):
                if self.query.iloc[i]['image_path']==matched.iloc[j]['image_path']:
                    continue
                assert self.query.iloc[i]['image_path'].split('/')[-2] == matched.iloc[j]['image_path'].split('/')[-2]
                self.pairs.append([self.query.iloc[i]['image_path'], matched.iloc[j]['image_path']])
                self.labels.append(1)
            # Keep 100 from different classes:
            unmatched = self.database.loc[self.database['id']!=id].sample(100, random_state=RANDOM_STATE)
            for j in range(len(unmatched)):
                assert self.query.iloc[i]['image_path'].split('/')[-2] + '_' \
                    + self.query.iloc[i]['image_path'].split('/')[-3] != unmatched.iloc[j]['image_path'].split('/')[-2] + '_' \
                        + unmatched.iloc[j]['image_path'].split('/')[-3]
                self.pairs.append([self.query.iloc[i]['image_path'], unmatched.iloc[j]['image_path']])
                self.labels.append(0)
        print()

    def __getitem__(self, index):

        # Open images
        image1 = Image.open(self.pairs[index][1])
        image2 = Image.open(self.pairs[index][0])

        label = self.labels[index]

        # Apply transform if any
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label

    def __len__(self):
        return len(self.pairs)


def calc_ssim_matrix(csv_file, transform=None, batch_size=32):
    df = pd.read_csv(csv_file)
    num_images = len(df)

    ssim_matrix = torch.zeros((num_images, num_images))#.to('cuda')

    # Calculate the number of batches
    num_batches = int(math.ceil(num_images / batch_size))

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min(num_images, (batch_index + 1) * batch_size)

        batch_images = []
        for i in range(start_index, end_index):
            image_path = df.iloc[i]['image_path']
            image = Image.open(image_path)
            if transform:
                image = transform(image)
            batch_images.append(image)#.to('cuda'))

        # Calculate pairwise SSIM values within the batch
        for i, image1 in enumerate(batch_images):
            image1 = image1.unsqueeze(0)
            for j in range(i+1, len(batch_images)):
                image2 = batch_images[j].unsqueeze(0)
                ssim_val = pytorch_msssim.ssim(image1, image2, data_range=1.0, size_average=True)
                ssim_matrix[start_index + i][start_index + j] = ssim_val
                ssim_matrix[start_index + j][start_index + i] = ssim_val

        print("Batch {}/{} completed".format(batch_index+1, num_batches))

    return ssim_matrix


class DynamicSamplingDataset(Dataset):
    def __init__(self, csv_file, ssim_matrix, transformA = None, transformB = None, hard_mining_ratio=0.5):
        self.csv_file = csv_file
        self.ssim_matrix = ssim_matrix
        self.hard_mining_ratio = hard_mining_ratio
        self.transformA = transformA
        self.transformB = transformB

        self.df = pd.read_csv(csv_file)
        self.unique_ids = self.df['id'].unique()
        self.num_unique_ids = len(self.unique_ids)
        # self.black_square_transform = transforms.Compose([transforms.Lambda(random_square_mask)])

    def __getitem__(self, index):
        random_float = torch.rand(1)

        if random_float<0.5:
            # Create postiive pair:
            # Select a random pair with the same class
            current_id = self.df.iloc[index]['id']
            samples = self.df[self.df['id'] == current_id]
            # Select a positive pair with the same class
            image1_path = self.df.iloc[index]['image_path']
            image2_path = samples.sample(n=1)['image_path'].tolist()[0]
            label = 1

        else:
            # Select a random pair with the same class
            current_id = self.df.iloc[index]['id']
            samples = self.df[self.df['id'] != current_id]
            # Select a positive pair with the same class
            image1_path = self.df.iloc[index]['image_path']
            image2_path = samples.sample(n=1)['image_path'].tolist()[0]
            label = 0

        # Load and transform the images
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        if self.transformA is not None:
            image1 = self.transformA(image1)
        if self.transformB is not None:
            image2 = self.transformB(image2)

        return image1, image2, label

    def __len__(self):
        return len(self.df)

class DynamicSamplingDataset2(Dataset):
    def __init__(self, csv_file, ssim_matrix, transformA = None, transformB = None, hard_mining_ratio=0.5):
        self.csv_file = csv_file
        self.ssim_matrix = ssim_matrix
        self.hard_mining_ratio = hard_mining_ratio
        self.transformA = transformA
        self.transformB = transformB

        self.df = pd.read_csv(csv_file)
        self.unique_ids = self.df['id'].unique()
        self.num_unique_ids = len(self.unique_ids)

    def __getitem__(self, index):
        # Select a random pair with the same class
        current_id = self.df.iloc[index]['id']
        samples = self.df[self.df['id'] == current_id]
        # Select a positive pair with the same class
        image1_path = self.df.iloc[index]['image_path']
        image2_path = samples.sample(n=1)['image_path'].tolist()[0]
        label = 1

        # Load and transform the images
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        if self.transformA is not None:
            image1 = self.transformA(image1)
        if self.transformB is not None:
            image2 = self.transformB(image2)

        if random()<=0.5:
            image1 = transforms.functional.vflip(image1)
            image2 = transforms.functional.vflip(image2)

        if random()<=0.5:
            return image1, image2, label, current_id
        else:
            return image2, image1, label, current_id


    def __len__(self):
        return len(self.df)

class DynamicSamplingDatasetnew(Dataset):
    def __init__(self, csv_file, ssim_matrix=None, transformA=None, transformB=None):
        self.csv_file = csv_file
        self.ssim_matrix = ssim_matrix
        self.transformA = transformA
        self.transformB = transformB

        # Load the CSV and group by ID
        self.df = pd.read_csv(csv_file)
        self.unique_ids = self.df['id'].unique()
        self.num_unique_ids = len(self.unique_ids)

        # Precompute all unique positive pairs, treating (pos1, pos2) the same as (pos2, pos1)
        self.pairs = []
        for unique_id in self.unique_ids:
            samples = self.df[self.df['id'] == unique_id]['image_path'].tolist()
            # Generate all combinations and enforce order
            self.pairs.extend([tuple(sorted(pair)) for pair in combinations(samples, 2)])

        # Remove duplicate pairs (enforced by sorting)
        self.pairs = list(set(self.pairs))
        self.labels = [1] * len(self.pairs)  # All positive pairs have a label of 1

        # Double the dataset by appending pairs with vertical flip
        self.pairs.extend(self.pairs)  # Duplicate the pairs
        self.labels.extend(self.labels)  # Duplicate the labels

        self.original_length = len(self.pairs) // 2  # Length of the original dataset


    def __getitem__(self, index):
        # Retrieve the image paths and label
        image1_path, image2_path = self.pairs[index]
        label = self.labels[index]

        # Load the images
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # Apply transformations if provided
        if self.transformA is not None:
            image1 = self.transformA(image1)
        if self.transformB is not None:
            image2 = self.transformB(image2)

        # Apply vertical flip to the second half of the dataset
        if index >= self.original_length:
            image1 = transforms.functional.vflip(image1)
            image2 = transforms.functional.vflip(image2)

        return image1, image2, 1, label

    def __len__(self):
        return len(self.pairs)

class DynamicSamplingDataset3(Dataset):
    def __init__(self, csv_file, ssim_matrix, transformA = None, transformB = None, hard_mining_ratio=0.5):
        self.csv_file = csv_file
        self.ssim_matrix = ssim_matrix
        self.hard_mining_ratio = hard_mining_ratio
        self.transformA = transformA
        self.transformB = transformB

        self.df = pd.read_csv(csv_file)
        self.unique_ids = self.df['id'].unique()
        self.num_unique_ids = len(self.unique_ids)

        # Initialize the list that will hold the new data
        pairs = []

        # Group the dataframe by 'id' and iterate over each group
        for group_id, group_df in self.df.groupby('id'):
            # Generate all unique combinations of image paths within this group
            img_pairs = list(combinations(group_df['image_path'], 2))

            # For each unique pair, add three entries to the new data list
            for img_path_1, img_path_2 in img_pairs:
                for flag in [-1, 0, 1]:
                    pairs.append({
                        'img_path_1': img_path_1,
                        'img_path_2': img_path_2,
                        'id': group_id,
                        'flag': flag
                    })

        # Create a new DataFrame with the new data
        self.df_pairs = pd.DataFrame(pairs)
        

    def __getitem__(self, index):
        # Select a random pair with the same class
        # current_id = self.df_pairs.iloc[index]['id']
        image1_path = self.df_pairs.iloc[index]['img_path_1']
        image2_path = self.df_pairs.iloc[index]['img_path_2']
        flag = self.df_pairs.iloc[index]['flag']
        label = 1

        # Load and transform the images
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        if flag == 0: 
            if self.transformA is not None:
                image1 = self.transformA(image1)
            if self.transformB is not None:
                image2 = self.transformB(image2)
        
        elif flag == -1:
            if self.transformA is not None:
                image1 = self.transformA(image1)
            image2 = transforms.ToTensor()(image2)

        elif flag == 1:
            if self.transformB is not None:
                image2 = self.transformB(image2)
            image1 = transforms.ToTensor()(image1)
        
        else:
            raise ValueError ("!!!Unknown flag!!!")

        return image1, image2, label

    def __len__(self):
        return len(self.df_pairs)

class SiameseDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        The SiameseDataset class is a PyTorch Dataset that generates pairs of images with corresponding labels
        for use in a Siamese network. The pairs consist of a positive example (images of the same identity)
        and randomly sample negative examples (images of different identities).
        :param csv_file: (str) Path to the CSV file containing image paths and labels.
        :param transform: (callable, optional) A function/transform that takes in an PIL image and returns a
        transformed version of it.
        """
        self.csv_file = csv_file
        self.transform = transform
        self.pairs, self.labels = self._load_pairs_and_labels()

    def _load_pairs_and_labels(self):
        """
        Load pairs and labels from CSV file.
        :return: (tuple) A tuple containing pairs and labels.
        """
        # Load CSV file as a pandas DataFrame
        df = pd.read_csv(self.csv_file)

        # Get unique identities
        unique_identities = df['id'].unique()

        pairs = []
        labels = []

        # Iterate through unique identities and create pairs
        for identity in unique_identities:
            # Select all samples with the same identity
            samples = df[df['id'] == identity]

            # Select samples with different identity for negative pairs
            negative_samples = df[df['id'] != identity]

            # Get the number of samples
            num_samples = len(samples)

            # Skip identities with only one sample
            if num_samples < 2:
                continue

            # Generate positive pairs (same identity)
            positive_pairs = [x + (1,) for x in combinations(samples['image_path'], 2)]

            # Generate negative pairs (different identity)
            negative_pairs = [(positive_pairs[i][0], negative_samples.sample(n=1, random_state=121)['image_path'].iloc[0], 0)
                              for i in range(len(positive_pairs))]

            # Add pairs and labels to the overall list
            pairs.extend(positive_pairs + negative_pairs)
            labels.extend([pair[2] for pair in positive_pairs + negative_pairs])

        return pairs, labels

    def __getitem__(self, index):
        """
        Get a pair of images and their label from the dataset.
        :param index: (int) Index of the pair of images.
        :return: (tuple) A tuple containing the pair of images and their label.
        """
        pair = self.pairs[index]
        label = self.labels[index]

        # Open images
        image1 = Image.open(pair[0])
        image2 = Image.open(pair[1])

        # Apply transform if any
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Return the pair of images and their label
        return image1.detach(), image2.detach(), label

    def __len__(self):
        """
        Get the length of the dataset.
        :return: (int) Length of the dataset.
        """
        return len(self.pairs)


class ImageBatchDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        """
        self.csv_file = csv_file
        self.transform = transform
        self.image_paths = self.csv_file['image_path'].tolist()

    def __getitem__(self, index):
        """
        Get a pair of images and their label from the dataset.
        :param index: (int) Index of the pair of images.
        :return: (tuple) A tuple containing the pair of images and their label.
        """
        img_path = self.image_paths[index]

        # Open image
        image1 = Image.open(img_path)

        # Apply transform if any
        if self.transform is not None:
            image1 = self.transform(image1)

        # Return the pair of images and their label
        return image1

    def __len__(self):
        """
        Get the length of the dataset.
        :return: (int) Length of the dataset.
        """
        return len(self.image_paths)


def random_square_mask(imgs, dimension_percentage=0.15, prob=0.5):
    img1, img2 = imgs  # unpack the tuple

    if torch.rand(1) < prob:
        h, w = img1.shape[1:]  # assuming image shape is (C, H, W)

        # Determine dimensions of the mask
        mask_h = int(h * dimension_percentage)
        mask_w = int(w * dimension_percentage)

        # Determine top left corner of the mask
        top = torch.randint(0, h - mask_h, (1,)).item()
        left = torch.randint(0, w - mask_w, (1,)).item()

        mask = torch.zeros((1, h, w))
        mask[:, top:top + mask_h, left:left + mask_w] = 1

        # Apply the mask to the images, assuming the image is in [0, 1]
        img1 = img1 - img1 * mask
        img2 = img2 - img2 * mask

    return img1, img2

def make_square_and_resize_tensor(img_tensor, fill_color=0, target_size=(224, 224)):
    batch_size, channels, height, width = img_tensor.size()
    size = max(height, width)
    h_padding = (size - width) // 2
    v_padding = (size - height) // 2

    # Padding to make the image square
    img_tensor = F.pad(img_tensor, (h_padding, h_padding, v_padding, v_padding), "constant", fill_color)

    # Resize to target size
    img_tensor = F.interpolate(img_tensor, size=target_size, mode="bilinear", align_corners=False)

    return img_tensor

def normalize_01(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    return (tensor - tensor_min) / (tensor_max - tensor_min)

gray_normalize = transforms.Normalize(
    # mean=[0.1019, 0.1019, 0.1019],
    # std=[0.1613, 0.1613, 0.1613]
    mean = [np.mean([0.485, 0.456, 0.406]), np.mean([0.485, 0.456, 0.406]), np.mean([0.485, 0.456, 0.406])],
    std = [np.mean([0.229, 0.224, 0.225]), np.mean([0.229, 0.224, 0.225]), np.mean([0.229, 0.224, 0.225])]
)

def find_best_thresholds(distances, similarities, ground_truth):
    assert len(distances) == len(similarities) == len(ground_truth), "Input lists must have the same length"

    # Convert lists to numpy arrays
    distances = np.array(distances)
    similarities = np.array(similarities)
    ground_truth = np.array(ground_truth)

    # Generate candidate thresholds
    distance_thresholds = np.linspace(np.min(distances), np.max(distances), num=200)
    similarity_thresholds = np.linspace(np.min(similarities), np.max(similarities), num=200)

    best_accuracy = 0
    best_distance_threshold = None
    best_similarity_threshold = None

    # Iterate over all possible combinations of distance and similarity thresholds
    for d_thr, s_thr in tqdm(product(distance_thresholds, similarity_thresholds)):
        # Predict labels based on thresholds
        predicted_labels = np.logical_or(distances <= d_thr, similarities >= s_thr).astype(int)

        # Calculate the accuracy
        accuracy = accuracy_score(ground_truth, predicted_labels)#accuracy_score(ground_truth, predicted_labels)

        # Update the best thresholds if the current accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_distance_threshold = d_thr
            best_similarity_threshold = s_thr

    return best_distance_threshold, best_similarity_threshold, best_accuracy

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function using Cosine Similarity.
    """

    def __init__(self, margin=0.4):
        super().__init__()
        self.margin = margin

    def forward(self, output, label):
        cosine_similarity = F.cosine_similarity(output[0], output[1], dim=1, eps=1e-6)

        # Cosine similarity ranges from -1 to 1, where -1 means exactly opposite, 0 means orthogonal (not similar nor dissimilar), and 1 means exactly the same.
        # However, we want "1" for similar and "0" for dissimilar, hence we add 1 to the cosine similarity and divide by 2 to re-scale it to [0, 1].
        cosine_similarity = (cosine_similarity + 1) / 2

        # For similar pairs, maximize cosine similarity
        pos_loss = (label) * (1 - cosine_similarity)**2

        # For dissimilar pairs, minimize cosine similarity with a margin (Yan Le Cun: margin - eucl_dist)
        neg_loss = (1 - label) * F.relu(self.margin - (1 - cosine_similarity))**2

        loss_contrastive = torch.mean(pos_loss + neg_loss)

        return loss_contrastive

def apply_transform_to_batch(batch):
    # Initialize an empty list to hold the transformed images
    transformed_images = []
    this_transform = transforms.Compose([transforms.Lambda(lambda x: x.repeat(3, 1, 1)),\
                                         gray_normalize,\
                                         transforms.Lambda(lambda x: make_square_and_resize_tensor(x))])
    # Loop over all images in the batch
    for image in batch:
        # Apply the transform to the image and append to the list
        transformed_images.append(this_transform(image))

    # Stack the list of images into a single tensor along the first dimension
    batch = torch.stack(transformed_images)

    return batch

def make_square_and_resize_tensor(tensor, fill_color=0, final_size=224):
    """
    Makes a tensor square by padding it with fill_color, then resizes it.

    Assumes tensor is a PyTorch tensor with shape (channels, height, width) and
    pixel values between 0 and 1.
    """
    # Note: PyTorch tensors use (channels, height, width) format

    sizes = tensor.size()
    y = sizes[-2]
    x = sizes[-1]
    size = max(x, y)

    # Normalize the fill color to [0,1]
    fill_color = fill_color / 255.

    # Padding values
    pad_left = pad_right = (size - x) // 2
    pad_top = pad_bottom = (size - y) // 2

    # If the padding on any side needs to be one pixel larger
    if (size - x) % 2 != 0:
        pad_right += 1
    if (size - y) % 2 != 0:
        pad_bottom += 1

    # Pad the tensor
    tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=fill_color)

    if len(tensor.size())<4:
        # Resize the tensor
        tensor = F.interpolate(tensor.unsqueeze(0), size=(final_size, final_size), mode='bilinear', align_corners=False)
    else:
        tensor = F.interpolate(tensor, size=(final_size, final_size), mode='bilinear', align_corners=False)
    return tensor.squeeze(0)

def make_square_and_resize(im, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(x, y)

    # Check the color mode of the input image
    if im.mode == 'RGB':
        fill_color = fill_color  # color image
    else:
        fill_color = fill_color[0]  # grayscale image

    # Create new image with appropriate color mode
    new_im = Image.new(im.mode, (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    new_im = new_im.resize((224, 224))
    return new_im

def faiss_inf(model, base_dataloader, query_dataloader, emb_size = 512):
    with torch.no_grad():
        model.eval()

        base_embeddings_list = []
        base_labels_list = []
        query_embeddings_list = []
        query_labels_list = []

        for _, data in enumerate(tqdm(base_dataloader)):
            img1, img2, label1, label2 = data
            img1, img2, label1, label2 = img1.cuda(), img2.cuda(), label1, label2
            with torch.no_grad():
                p1, p2, z1, z2, _, _ = model(x1 = img1, x2 = img2)
                base_embeddings_list.extend(p1.cpu().detach().numpy())
                base_labels_list.extend(label1.cpu().detach().numpy())
                base_embeddings_list.extend(p2.cpu().detach().numpy())
                base_labels_list.extend(label2.cpu().detach().numpy())
        base_embeddings = np.asarray(base_embeddings_list)
        base_labels = np.asarray(base_labels_list)

        for batch_idx, data in enumerate(tqdm(query_dataloader)):
            img1, img2, label1, label2 = data
            img1, img2, label1, label2 = img1.cuda(), img2.cuda(), label1, label2
            with torch.no_grad():
                p1, p2, z1, z2, _, _ = model(x1 = img1, x2 = img2)
                query_embeddings_list.extend(p1.cpu().detach().numpy())
                query_labels_list.extend(label1.cpu().detach().numpy())
                query_embeddings_list.extend(p2.cpu().detach().numpy())
                query_labels_list.extend(label2.cpu().detach().numpy())
        query_embeddings = np.asarray(query_embeddings_list)
        query_labels = np.asarray(query_labels_list)

        # faiss reference for cosine simalirity:
        # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
        index = faiss.IndexFlatIP(emb_size)

        # L2 normalization of base embeddings so that using dot product on normalized embeddings we end up with cosine similarity:
        faiss.normalize_L2(base_embeddings)
        index.add(base_embeddings)

        # L2 normalization of query embeddings so that using dot product on normalized embeddings we end up with cosine similarity:
        faiss.normalize_L2(query_embeddings)
        cosine_similarity, idxs = index.search(query_embeddings, len(base_labels_list))
        # print('Cosine Similarity by FAISS:{}'.format(cosine_similarity))
        
        cnt=0
        labels = []
        cos_sim = []
        for i in range(np.shape(cosine_similarity)[0]):
            # print('#########################################################')
            for j in range(np.shape(cosine_similarity)[1]):
                if base_labels[idxs[i][j]] == query_labels[i]:
                    # print(query_labels[i], base_labels[idxs[i][j]], 1, cosine_similarity[i][j])
                    labels.append(1)
                    cos_sim.append(cosine_similarity[i][j])
                else:
                    cnt+=1
                    # print(query_labels[i], base_labels[idxs[i][j]], 0, cosine_similarity[i][j])
                    labels.append(0)
                    cos_sim.append(cosine_similarity[i][j])
        
        eer_loss = compute_eer(labels, cos_sim)
        # print('Test Loss: {:.4f}'.format(eer_loss))

        # del labels
        # del cos_sim
        # del idxs
        # del cosine_similarity
        

        # # Top-8 for weights calculation:
        # cosine_similarity, idxs = index.search(query_embeddings, 8)
        # erroneous_ids = {}
        # cnt=0
        # for i in range(np.shape(cosine_similarity)[0]):
        #     # print('#########################################################')
        #     for j in range(np.shape(cosine_similarity)[1]):
        #         if base_labels[idxs[i][j]] != query_labels[i]:
        #             erroneous_id = base_labels[idxs[i][j]]
        #             erroneous_ids[erroneous_id] = erroneous_ids.get(erroneous_id, 0) + 1
        #             erroneous_id = query_labels[i]
        #             erroneous_ids[erroneous_id] = erroneous_ids.get(erroneous_id, 0) + 1
                    

        del index
        del base_embeddings_list
        del base_labels_list
        del query_embeddings_list
        del query_labels_list

        return eer_loss#, erroneous_ids

def faiss_inf_ow(model, base_dataloader, query_dataloader, emb_size = 512):
    with torch.no_grad():
        model.eval()
        base_embeddings_list = []
        base_labels_list = []
        query_embeddings_list = []
        query_labels_list = []
        for _, data in enumerate(tqdm(base_dataloader)):
            img1, img2, label1, label2 = data
            img1, img2, label1, label2 = img1.cuda(), img2.cuda(), label1, label2
            with torch.no_grad():
                p1, p2, z1, z2, _, _ = model(x1 = img1, x2 = img2)
                # p1, p2, z1, z2 = model(x1 = img1, x2 = img2)
                # Half id occurences from 1 branch and half id occurences from barnch 2:
                base_embeddings_list.extend(p1.cpu().detach().numpy()[::2])
                base_embeddings_list.extend(p2.cpu().detach().numpy()[::2])
                base_labels_list.extend(label1.cpu().detach().numpy()[::2])
                base_labels_list.extend(label2.cpu().detach().numpy()[::2])
                query_embeddings_list.extend(p1.cpu().detach().numpy()[1::2])
                query_embeddings_list.extend(p2.cpu().detach().numpy()[1::2])
                query_labels_list.extend(label1.cpu().detach().numpy()[1::2])
                query_labels_list.extend(label2.cpu().detach().numpy()[1::2])
        base_embeddings = np.asarray(base_embeddings_list)
        base_labels = np.asarray(base_labels_list)
                
        query_embeddings = np.asarray(query_embeddings_list)
        query_labels = np.asarray(query_labels_list)
        # faiss reference for cosine simalirity:
        # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
        index = faiss.IndexFlatIP(emb_size)
        # L2 normalization of base embeddings so that using dot product on normalized embeddings we end up with cosine similarity:
        faiss.normalize_L2(base_embeddings)
        index.add(base_embeddings)
        # L2 normalization of query embeddings so that using dot product on normalized embeddings we end up with cosine similarity:
        faiss.normalize_L2(query_embeddings)
        cosine_similarity, idxs = index.search(query_embeddings, len(base_labels_list))
        # print('Cosine Similarity by FAISS:{}'.format(cosine_similarity))
        
        cnt=0
        labels = []
        cos_sim = []
        for i in range(np.shape(cosine_similarity)[0]):
            # print('#########################################################')
            for j in range(np.shape(cosine_similarity)[1]):
                if base_labels[idxs[i][j]] == query_labels[i]:
                    # print(query_labels[i], base_labels[idxs[i][j]], 1, cosine_similarity[i][j])
                    labels.append(1)
                    cos_sim.append(cosine_similarity[i][j])
                else:
                    cnt+=1
                    # print(query_labels[i], base_labels[idxs[i][j]], 0, cosine_similarity[i][j])
                    labels.append(0)
                    cos_sim.append(cosine_similarity[i][j])
        
        eer_loss = compute_eer(labels, cos_sim)
                    
        del index
        del base_embeddings_list
        del base_labels_list
        
        return eer_loss

def init_uniform(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -0.1, 0.1)

def init_normal(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean=0.0, std=0.1) 

def init_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def init_kaiming(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def init_orthogonal(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def compute_roc_Ou(embeddings, targets, class_num):
    """
    SOS: it first performs a l2-normalization that allows the use of matmul to calculate cosine similarity
    """
    embeddings = preprocessing.normalize(embeddings)
    emb_num = len(embeddings)
    # Cosine similarity between any two pairs, note that all embeddings are l2-normalized
    scores = np.matmul(embeddings, embeddings.T)
    samples_per_class = emb_num // class_num
    # define matching pairs
    intra_class_combinations = np.array(list(combinations(range(samples_per_class), 2)))
    match_pairs = [i*samples_per_class + intra_class_combinations for i in range(class_num)]
    match_pairs = np.concatenate(match_pairs, axis=0)
    scores_match = scores[match_pairs[:, 0], match_pairs[:, 1]]
    labels_match = np.ones(len(match_pairs))

    # define imposter pairs
    inter_class_combinations = np.array(list(combinations(range(class_num), 2)))
    imposter_pairs = [np.expand_dims(i*samples_per_class, axis=0) for i in inter_class_combinations]
    imposter_pairs = np.concatenate(imposter_pairs, axis=0)
    scores_imposter = scores[imposter_pairs[:, 0], imposter_pairs[:, 1]]
    labels_imposter = np.zeros(len(imposter_pairs))

    # merge matching pairs and imposter pairs and assign labels
    all_scores = np.concatenate((scores_match, scores_imposter))
    all_labels = np.concatenate((labels_match, labels_imposter))
    # compute roc, auc and eer
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    return fpr, tpr, thresholds, scores_match, scores_imposter, embeddings, targets

def compute_eer_Ou(fpr, tpr):
    fnr = 1 - tpr
    # find indices where EER, fpr100, fpr1000, fpr0, best acc occur
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))

    # compute EER, FRR@FAR=0.01, FRR@FAR=0.001, FRR@FAR=0
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return eer


def calculate_feature_entropy(embedding):
    nbins = 10
    hist_range = (embedding.min(), embedding.max())
    hist = torch.histc(embedding, bins=nbins, min=hist_range[0].cpu().detach().numpy().tolist(), max=hist_range[1].cpu().detach().numpy().tolist())
    prob = hist / torch.sum(hist)
    prob = prob[prob > 0]  # Remove zeros for stability in log computation
    return -torch.sum(prob * torch.log(prob))

def feature_range_loss(p):
    return -torch.mean(torch.max(p, dim=0)[0] - torch.min(p, dim=0)[0])

def normalized_activation_variance_loss(p):
    normalized_p = (p - p.mean(dim=0)) / p.std(dim=0)
    return -torch.mean(normalized_p.var(dim=0))

# def inverse_activation_frequency_loss(p):
#     activation_counts = torch.sum(p > 0.5, dim=0)
#     # activation_counts = torch.sum(torch.abs(p) > 0.01, dim=0)
#     return torch.mean(1.0 / (activation_counts + 1e-5))

# def inverse_activation_frequency_loss(embeddings, epsilon=1e-5):

#     # Count how many times each neuron is activated across the batch
#     activation_counts = torch.sum(embeddings > 0, dim=0).float()

#     # Calculate the frequency of activation for each neuron
#     activation_freq = activation_counts / (embeddings.size(0) + epsilon)

#     # Normalize frequencies to sum to 1 (like a probability distribution)
#     activation_freq = activation_freq / torch.sum(activation_freq)

#     # Calculate entropy
#     entropy = -torch.sum(activation_freq * torch.log(activation_freq + epsilon))  # Add epsilon for numerical stability

#     return entropy


def activation_entropy_loss(p):
    activation_prob = torch.softmax(p, dim=0)
    return torch.mean(-torch.sum(activation_prob * torch.log(activation_prob + 1e-5), dim=0))

def peak_to_mean_activation_ratio_loss(p):
    return -torch.mean(torch.max(p, dim=0)[0] / (torch.mean(p, dim=0) + 1e-5))

def cov_loss(embedding):
    cov_matrix = torch.cov(embedding.T)
    off_diagonal_elements = cov_matrix - torch.diag(cov_matrix.diag())
    return off_diagonal_elements.abs().mean()

def feature_decorrelation_loss(embedding):
    # Calculate the covariance matrix
    cov_matrix = torch.cov(embedding.T)
    # Penalize non-zero off-diagonal elements
    loss = torch.mean(offdiag_elements(cov_matrix) ** 2)
    return loss

def offdiag_elements(matrix):
    # Create a mask for the off-diagonal elements
    mask = torch.ones_like(matrix).bool()
    mask.fill_diagonal_(0)
    return matrix[mask]

def orthogonal_regularization(embedding):
    # Normalize the embeddings
    normalized_embeddings = F.normalize(embedding, p=2, dim=1)
    gram_matrix = torch.mm(normalized_embeddings.T, normalized_embeddings)
    identity = torch.eye(gram_matrix.shape[1]).to(gram_matrix.device)
    return ((gram_matrix - identity) ** 2).mean()

def std_invariance_loss(embedding_x, embedding_y):
    embedding_x = embedding_x - embedding_x.mean(dim=0)
    embedding_y = embedding_y - embedding_y.mean(dim=0)
    std_x = torch.sqrt(embedding_x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(embedding_y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    return std_loss

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss


def custom_loss(embeddings, labels, n_classes = 512):
    
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

    # n_classes = labels.unique().numel()
    n_samples, n_features = embeddings.size()

    labels = labels[:,None]

    mask = torch.eq(labels, labels.t()).bool()

    cov_matrix = torch.cov(embeddings)
    
    # Covariance from pairs of same classes
    cov_same_classes = cov_matrix[mask].mean()

    # Covariance from pairs of different classes
    cov_diff_classes = cov_matrix[~mask].mean()

    # Gram matrix for orthogonality:
    gram_matrix = torch.mm(embeddings, embeddings.T)
    orthogonality_loss = torch.norm(gram_matrix[~mask], p='fro')
    total_loss = (1 - cov_same_classes) + cov_diff_classes + orthogonality_loss
    # # Mean embeddings per class
    # mean_embeddings = torch.mm(masks.T, embeddings) / masks.sum(0).unsqueeze(1)

    # # Center embeddings within each class
    # centered_embeddings = embeddings - torch.mm(masks, mean_embeddings)

    # # Covariance within classes
    # within_class_cov = torch.bmm(centered_embeddings.unsqueeze(2), centered_embeddings.unsqueeze(1)).mean()

    # # Orthogonality across classes
    # mean_embeddings_expanded = mean_embeddings.unsqueeze(0).expand(n_classes, n_classes, n_features)
    # orthogonality_loss = (mean_embeddings_expanded * mean_embeddings_expanded.transpose(0, 1)).sum(dim=2).triu(1).sum()

    # # Combine the losses
    # total_loss = within_class_cov + orthogonality_loss

    return total_loss

def ortho_loss(embeddings, labels):
    labels = labels[:,None]
    mask = torch.eq(labels, labels.t()).bool()
    # Gram matrix for orthogonality:
    gram_matrix = torch.mm(embeddings, embeddings.T)
    identity = torch.eye(embeddings.shape[0]).to(gram_matrix.device)
    # orthogonality_loss = torch.norm(gram_matrix[~mask], p='fro')
    return ((gram_matrix[~mask] - identity[~mask]) ** 2).mean()
    
def orthogonality_loss(weights):
    gram_matrix = torch.mm(weights, weights.t())
    gram_matrix.fill_diagonal_(0)
    loss = torch.norm(gram_matrix, p='fro')
    return loss

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

def minimum_entropy_regularization_loss(p):
    # Normalize activations to probabilities
    p_normalized = torch.softmax(p, dim=0)
    entropy = -torch.sum(p_normalized * torch.log(p_normalized + 1e-5), dim=0)

    # Penalize high entropy
    loss = torch.mean(entropy)
    return loss


def adaptive_geometric_regularization_loss(embeddings, num_zones=3):
    """
    Calculates an adaptive geometric regularization loss.

    Args:
    - embeddings (torch.Tensor): The embeddings tensor, size (batch_size, embedding_size).
    - num_zones (int): Number of geometric zones to divide the embedding space.

    Returns:
    - torch.Tensor: The adaptive geometric regularization loss.
    """
    # Normalize the embeddings to project onto a hypersphere
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

    # Calculate zone centroids
    zone_indices = torch.randint(0, num_zones, (embeddings.shape[0],))
    centroids = []
    for i in range(num_zones):
        zone_embeddings = normalized_embeddings[zone_indices == i]
        if len(zone_embeddings) > 0:
            centroids.append(zone_embeddings.mean(dim=0))
        else:
            centroids.append(torch.zeros_like(embeddings[0]))
    centroids = torch.stack(centroids)

    # Calculate loss based on deviation from centroids and centroid distribution
    centroid_distances = torch.cdist(centroids, centroids, p=2)
    intra_zone_loss = F.mse_loss(normalized_embeddings, centroids[zone_indices])
    inter_zone_loss = centroid_distances.mean()

    # Combine losses
    loss = intra_zone_loss + inter_zone_loss
    # print(loss)
    return loss
