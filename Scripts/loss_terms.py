
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot

class CosFace(nn.Module):
    def __init__(self, s=30.0, m=0.2):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, input, labels):
        # input: size = B x num_class
        cos = input
        one_hot = torch.zeros(cos.size()).cuda()
        one_hot = one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = self.s * (cos - one_hot * self.m)

        softmax_output = F.log_softmax(output, dim=1)
        loss = -1 * softmax_output.gather(1, labels.view(-1, 1))
        loss = loss.mean()

        return loss



class KLDivFeatureMapsLoss(nn.Module):
    def __init__(self, reduction="batchmean"):
        """
        KL Divergence between two feature maps.
        Args:
        - reduction: Specifies the reduction applied to the output ('batchmean', 'sum', or 'none').
                     'batchmean' computes the mean KL divergence across the batch.
        """
        super(KLDivFeatureMapsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, feature_map_1, feature_map_2):
        """
        Compute the KL divergence loss between two feature maps.
        
        Args:
        - feature_map_1: Feature map from branch 1, shape (B, C, H, W) or (B, D).
        - feature_map_2: Feature map from branch 2, shape (B, C, H, W) or (B, D).
        
        Returns:
        - loss: The KL divergence loss.
        """
        # Ensure both feature maps have the same shape
        assert feature_map_1.shape == feature_map_2.shape, "Feature maps must have the same shape!"

        # Flatten spatial dimensions if needed (e.g., (B, C, H, W) -> (B, C*H*W))
        if feature_map_1.dim() > 2:
            B, C, H, W = feature_map_1.shape
            feature_map_1 = feature_map_1.view(B, -1)
            feature_map_2 = feature_map_2.view(B, -1)

        # Normalize feature maps to probabilities (e.g., using softmax along the feature dimension)
        prob_map_1 = F.softmax(feature_map_1, dim=1)
        prob_map_2 = F.softmax(feature_map_2, dim=1)

        # Compute the KL divergence
        loss = F.kl_div(prob_map_1.log(), prob_map_2, reduction=self.reduction)
        return loss



class SymmetricKLDivLoss(nn.Module):
    def __init__(self):
        """
        Symmetric KL Divergence between feature maps.
        """
        super(SymmetricKLDivLoss, self).__init__()

    def forward(self, feature_map_1, feature_map_2):
        """
        Compute the symmetric KL divergence loss between two feature maps.

        Args:
        - feature_map_1: Feature map from branch 1, shape (B, C, H, W) or (B, D).
        - feature_map_2: Feature map from branch 2, shape (B, C, H, W) or (B, D).

        Returns:
        - loss: The symmetric KL divergence loss.
        """
        # Ensure feature maps have the same shape
        assert feature_map_1.shape == feature_map_2.shape, "Feature maps must have the same shape!"

        # Flatten spatial dimensions if needed
        if feature_map_1.dim() > 2:
            B, C, H, W = feature_map_1.shape
            feature_map_1 = feature_map_1.view(B, -1)
            feature_map_2 = feature_map_2.view(B, -1)

        # Normalize feature maps to probabilities
        prob_map_1 = F.softmax(feature_map_1, dim=1)
        prob_map_2 = F.softmax(feature_map_2, dim=1)

        # Compute forward KL divergence
        kl_1_to_2 = F.kl_div(prob_map_1.log(), prob_map_2, reduction="none").sum(dim=1)
        kl_2_to_1 = F.kl_div(prob_map_2.log(), prob_map_1, reduction="none").sum(dim=1)

        # Symmetric KL divergence
        sym_kl = 0.5 * (kl_1_to_2 + kl_2_to_1)

        # Mean over batch
        return sym_kl.mean()


class KLDivFeatureMapsLoss(nn.Module):
    def __init__(self, reduction="sum"):
        """
        KL Divergence between two feature maps (B x C x H x W).
        Args:
        - reduction: Specifies reduction applied to output ('batchmean', 'sum', 'none').
        """
        super(KLDivFeatureMapsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, feature_map_1, feature_map_2):
        """
        Compute KL divergence loss between two feature maps.
        
        Args:
        - feature_map_1: Feature map from branch 1, shape (B, C, H, W).
        - feature_map_2: Feature map from branch 2, shape (B, C, H, W).
        
        Returns:
        - loss: KL divergence loss.
        """
        # Ensure both feature maps have the same shape
        assert feature_map_1.shape == feature_map_2.shape, "Feature maps must have the same shape!"

        # Flatten the spatial dimensions (H, W) -> (H * W)
        B, C, H, W = feature_map_1.shape
        feature_map_1 = feature_map_1.view(B, C, -1)  # B x C x (H * W)
        feature_map_2 = feature_map_2.view(B, C, -1)  # B x C x (H * W)

        # Normalize to probabilities along spatial dimensions
        prob_map_1 = F.softmax(feature_map_1, dim=2)  # Normalize over H*W
        prob_map_2 = F.softmax(feature_map_2, dim=2)  # Normalize over H*W

        # Compute KL divergence along the spatial dimensions
        kl_div = F.kl_div(prob_map_1.log(), prob_map_2, reduction="none")  # B x C x (H * W)

        # Aggregate KL divergence across spatial dimensions
        kl_div = kl_div.sum(dim=2)  # B x C

        # Reduce over batch and channel dimensions
        if self.reduction == "batchmean":
            loss = kl_div.mean()  # Mean over B and C
        elif self.reduction == "sum":
            loss = kl_div.sum()  # Sum over B and C
        elif self.reduction == "none":
            loss = kl_div  # Return raw channel-wise KL divergence for each batch
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

        return loss



class CrossCovarianceLoss(nn.Module):
    def __init__(self):
        super(CrossCovarianceLoss, self).__init__()

    def forward(self, feature_map_1, feature_map_2):
        # Ensure feature maps are 2D (B, D)
        if feature_map_1.dim() == 4:  # If (B, C, H, W)
            feature_map_1 = torch.mean(feature_map_1, dim=[2, 3])  # Global Average Pooling
            feature_map_2 = torch.mean(feature_map_2, dim=[2, 3])

        # Normalize feature maps
        feature_map_1 = feature_map_1 - feature_map_1.mean(dim=0, keepdim=True)
        feature_map_2 = feature_map_2 - feature_map_2.mean(dim=0, keepdim=True)

        # Compute covariance matrices
        cov_1 = (feature_map_1.T @ feature_map_1) / (feature_map_1.shape[0] - 1)
        cov_2 = (feature_map_2.T @ feature_map_2) / (feature_map_2.shape[0] - 1)

        # Frobenius norm of covariance difference
        loss = torch.norm(cov_1 - cov_2, p='fro') ** 2
        return loss

def inverse_activation_frequency_loss(p):
    activation_counts = torch.sum(p > 0.05, dim=0)
    # activation_counts = torch.sum(torch.abs(p) > 0.01, dim=0)
    return torch.mean(1.0 / (activation_counts + 1e-5))

def compute_angular_variance(embeddings, labels):
    """
    Compute intra-class and inter-class angular variance.
    Args:
    - embeddings: Tensor of shape (B, D), where B is batch size, D is embedding dimension.
    - labels: Tensor of shape (B,), where each element is the class label.
    Returns:
    - intra_class_variance: Variance of angles within the same class.
    - inter_class_variance: Variance of angles across different classes.
    """
    # Normalize embeddings to unit vectors
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarity matrix
    cos_sim_matrix = torch.matmul(embeddings, embeddings.T)

    # Create mask for intra-class pairs
    label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    intra_class_sim = cos_sim_matrix[label_mask].view(-1)

    # Create mask for inter-class pairs
    inter_class_sim = cos_sim_matrix[~label_mask].view(-1)

    # Compute variance
    intra_class_variance = torch.var(intra_class_sim)
    inter_class_variance = torch.var(inter_class_sim)

    return -inter_class_variance + intra_class_variance


class PrototypeLearning(nn.Module):
    def __init__(self, num_classes, embedding_dim, margin=0.1, delta=0.2):
        super(PrototypeLearning, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim)).to('cuda')
        self.margin = margin
        self.delta = delta

    def forward(self, embeddings, labels):
        # Normalize embeddings and prototypes
        embeddings = F.normalize(embeddings, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        # Compute distances to prototypes
        distances = torch.cdist(embeddings, prototypes, p=2)  # Shape: (B, C)

        # Prototype-Alignment Loss
        labels_one_hot = F.one_hot(labels, num_classes=prototypes.size(0)).float()
        alignment_loss = (labels_one_hot * distances).sum(dim=1).mean()

        # Prototype Regularization Loss
        pairwise_distances = torch.cdist(prototypes, prototypes, p=2)
        reg_loss = F.relu(self.delta - pairwise_distances).mean()

        # Total loss
        total_loss = alignment_loss + self.margin * reg_loss
        return total_loss

class DirectionalContrastiveLoss(nn.Module):
    def __init__(self, tau=0.5):
        """
        Directional Contrastive Loss
        Args:
        - tau: Threshold for negative pair repulsion.
        """
        super(DirectionalContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, embeddings1, embeddings2, labels):
        """
        Args:
        - embeddings1: Tensor of shape (B, D), first embedding in the pair.
        - embeddings2: Tensor of shape (B, D), second embedding in the pair.
        - labels: Tensor of shape (B,), binary labels (1 for positive, 0 for negative).
        Returns:
        - loss: Directional Contrastive Loss.
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(embeddings1, embeddings2, dim=1)

        # Positive pairs: (1 - cos(theta))^2
        positive_loss = (1 - cosine_similarity) ** 2

        # Negative pairs: max(0, cos(theta) - tau)^2
        negative_loss = F.relu(cosine_similarity - self.tau) ** 2

        # Combine losses based on labels
        loss = labels * positive_loss + (1 - labels) * negative_loss
        return loss.mean()

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

import torch
import torch.nn as nn
import torch.nn.functional as F

class AngularMarginContrastiveLoss(nn.Module):
    def __init__(self, initial_s=30.0, initial_m=0.5):
        """
        Angular Margin Contrastive Loss with Learnable Scaling (AMCL-LS)
        Args:
        - initial_s (float): Initial value for the learnable scaling factor.
        - initial_m (float): Initial value for the learnable margin.
        """
        super(AngularMarginContrastiveLoss, self).__init__()
        self.s = nn.Parameter(torch.tensor(initial_s, dtype=torch.float32))  # Learnable scaling factor
        self.m = nn.Parameter(torch.tensor(initial_m, dtype=torch.float32))  # Learnable margin

    def forward(self, embeddings, labels):
        """
        Args:
        - embeddings: Tensor of shape (B, D), where B is batch size, D is embedding dimension.
        - labels: Tensor of shape (B,), containing class labels.
        Returns:
        - loss: Scalar loss value.
        """
        batch_size = embeddings.size(0)

        # Normalize embeddings to unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute cosine similarity matrix
        cos_sim = torch.matmul(embeddings, embeddings.T)  # (B, B)

        # Create masks for positives and negatives
        labels = labels.unsqueeze(1)  # (B, 1)
        pos_mask = labels.eq(labels.T).float()  # (B, B), 1 for positives
        neg_mask = 1.0 - pos_mask  # (B, B), 1 for negatives

        # Positive term: (1 - cosine similarity) scaled by learnable factor
        pos_term = pos_mask * (1 - cos_sim)

        # Negative term: max(0, cosine similarity - margin) scaled by learnable factor
        neg_term = neg_mask * F.relu(cos_sim - self.m)

        # Total loss
        loss = self.s * (pos_term.sum() + neg_term.sum()) / (batch_size * (batch_size - 1))
        return loss

class FrequencyLoss(nn.Module):
    def __init__(self, normalize=True, loss_type='kl'):
        super(FrequencyLoss, self).__init__()
        self.normalize = normalize
        self.loss_type = loss_type  # 'l2', 'cosine', 'kl', 'js'

    def fft_feature_map(self, feature_map):
        # Compute FFT magnitudes
        fft = torch.fft.fftn(feature_map, dim=(-2, -1))  # Apply 2D FFT along spatial dimensions
        fft_mag = torch.abs(fft)  # Magnitude only
        return fft_mag

    def forward(self, feature1, feature2):
        # Compute FFT magnitudes for both feature maps
        fft1 = self.fft_feature_map(feature1)
        fft2 = self.fft_feature_map(feature2)

        # Flatten and normalize if required
        if self.normalize:
            fft1 = F.normalize(fft1.flatten(1), dim=1)
            fft2 = F.normalize(fft2.flatten(1), dim=1)

        # Compute loss based on type
        if self.loss_type == 'l2':
            return torch.norm(fft1 - fft2, p=2)
        elif self.loss_type == 'cosine':
            return -torch.mean(torch.sum(fft1 * fft2, dim=1))  # Cosine similarity loss
        elif self.loss_type == 'kl':
            fft1 = F.log_softmax(fft1, dim=1)
            fft2 = F.softmax(fft2, dim=1)
            return F.kl_div(fft1, fft2, reduction='batchmean') # KL divergence
        elif self.loss_type == 'js':
            fft1 = F.softmax(fft1, dim=1)
            fft2 = F.softmax(fft2, dim=1)
            m = 0.5 * (fft1 + fft2)
            return 0.5 * (F.kl_div(F.log_softmax(fft1, dim=1), m, reduction='batchmean') +
                          F.kl_div(F.log_softmax(fft2, dim=1), m, reduction='batchmean'))  # JS divergence
        else:
            raise ValueError("Unsupported loss_type. Use 'l2', 'cosine', 'kl', or 'js'")

def emd_loss(mcal1, mcal2):
    """
    Computes Earth Mover's Distance (EMD) between two MCAL attention maps.
    Args:
        mcal1, mcal2: Tensor of shape (B, H, W) - MCAL attention maps for two inputs
    Returns:
        EMD loss (scalar)
    """
    mcal1 = mcal1.squeeze()
    mcal2 = mcal2.squeeze()
    # Get batch size, height, and width
    B, H, W = mcal1.shape

    # Flatten spatial dimensions (H, W) into a single dimension (H*W)
    mcal1 = mcal1.view(B, -1)  # Shape: (B, H*W)
    mcal2 = mcal2.view(B, -1)  # Shape: (B, H*W)

    # Normalize to create valid probability distributions (sum = 1)
    mcal1 = mcal1 / (mcal1.sum(dim=1, keepdim=True) + 1e-8)
    mcal2 = mcal2 / (mcal2.sum(dim=1, keepdim=True) + 1e-8)

    # Create spatial coordinate grid (Euclidean distances for cost matrix)
    device = mcal1.device
    coords = torch.stack(torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device), indexing='ij'
    ), dim=-1).view(-1, 2)  # Shape: (H*W, 2)

    # Compute pairwise Euclidean distances (cost matrix)
    cost_matrix = torch.cdist(coords.float(), coords.float(), p=2)  # Shape: (H*W, H*W)

    # Compute EMD for each batch
    loss = 0.0
    for i in range(B):
        loss += ot.emd2(
            mcal1[i].cpu().detach().numpy(),
            mcal2[i].cpu().detach().numpy(),
            cost_matrix.cpu().detach().numpy()
        )

    # Average loss across the batch
    return loss / B

class AdaptiveWeightedLoss(nn.Module):
    def __init__(self, alpha=3.0):
        """
        Args:
            alpha: Scaling factor for difficulty weighting.
        """
        super(AdaptiveWeightedLoss, self).__init__()
        self.alpha = alpha  # Controls scaling sensitivity

    def forward(self, embedding1, embedding2):
        # Compute base loss (cosine similarity)
        loss = 1 - F.cosine_similarity(embedding1, embedding2, dim=1)

        # Compute pair difficulty
        difficulty = loss.detach()  # Use cosine distance directly as difficulty

        # Scale loss based on difficulty
        weights = torch.pow(difficulty, self.alpha)  # Higher weights for harder pairs
        weighted_loss = weights * loss
        return weighted_loss.mean()

