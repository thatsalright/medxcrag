
import torch
import torch.nn.functional as F

def compute_contrastive_loss(similarity_matrix, labels=None, temperature=0.05, hard_negative_mask=None):

    batch_size = similarity_matrix.size(0)
    
    # If no label matrix provided, use diagonal as positive pairs
    if labels is None:
        labels = torch.eye(batch_size).to(similarity_matrix.device)
    
    # Normalize similarity matrix
    similarity_matrix = similarity_matrix / temperature
    
    # Calculate exp(similarity)
    exp_sim = torch.exp(similarity_matrix)
    
    # If using hard negative mining
    if hard_negative_mask is not None:
        # Check if hard_negative_mask is a tuple (mask, count)
        if isinstance(hard_negative_mask, tuple):
            hard_negative_mask = hard_negative_mask[0]  # Extract just the mask
            
        # Diagonal also marked as 1 (include positive pairs)
        diagonal_mask = torch.eye(batch_size).to(similarity_matrix.device)
        mask = diagonal_mask + hard_negative_mask
        
        # Set exp_sim to 0 for positions not in mask
        exp_sim = exp_sim * mask
    
    # Diagonal mask (for extracting positive pair scores)
    pos_mask = torch.eye(batch_size).to(similarity_matrix.device)
    
    # Calculate positive pair scores
    pos_sim = torch.sum(exp_sim * pos_mask, dim=1)
    
    # Calculate sum of all pair scores
    all_sim = torch.sum(exp_sim, dim=1)
    
    # Calculate negative log likelihood loss
    eps = 1e-8  # Avoid division by zero
    loss = -torch.mean(torch.log((pos_sim + eps) / (all_sim + eps)))
    
    return loss

def compute_bidirectional_contrastive_loss(similarity_matrix, temperature=0.05, hard_negative_mask=None):

    # Image-to-text loss
    i2t_loss = compute_contrastive_loss(similarity_matrix, None, temperature, hard_negative_mask)
    
    # Text-to-image loss (transpose similarity matrix)
    if hard_negative_mask is not None:
        if isinstance(hard_negative_mask, tuple):
            mask = hard_negative_mask[0]
            t2i_mask = (mask.t(), mask.sum().item())
        else:
            t2i_mask = hard_negative_mask.t()
    else:
        t2i_mask = None
        
    t2i_loss = compute_contrastive_loss(similarity_matrix.t(), None, temperature, t2i_mask)
    
    # Bidirectional loss
    loss = (i2t_loss + t2i_loss) / 2.0
    
    return loss