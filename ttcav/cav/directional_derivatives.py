import torch
from ttcav.data_processing.data_loader import get_dataloader
from ttcav.cav.utils import get_activations_grads_attentions
import os
import json

def get_attn_directional_derivatives(cavs, grads, attentions, num_cavs, device):
    """Input: CAVs, gradients and attentions associated with a specific layer."""

    all_directional_derivatives = []

    avg_attn_head = attentions[:, :, 0, :].mean(dim=1).to(device) # shape (batch, n_tokens) 
    avg_attn_head_norms = avg_attn_head.norm(p=2, dim=1, keepdim=True).to(device)  

    for cav_idx in range(num_cavs):

        if num_cavs == 1:
            cav = cavs
        else:    
            cav = cavs[cav_idx]
        
        directional_derivative = torch.matmul(grads.float().to(device), cav.float().to(device)) # shape (batch, n_tokens)

        directional_derivative_norms = directional_derivative.norm(p=2, dim=1, keepdim=True).to(device)  
        directional_derivatives_scaled = directional_derivative * (avg_attn_head_norms / (directional_derivative_norms) + 1e-8) # Avoid division by zero
        attn_directional_derivatives = avg_attn_head.squeeze(0) * directional_derivatives_scaled.squeeze(0)

        if num_cavs == 1:
            return attn_directional_derivatives
                
        all_directional_derivatives.append(attn_directional_derivatives) 
    
    return torch.stack(all_directional_derivatives, dim=-1) # shape (batch, n_tokens, num_cavs)



def heatmap_attn_directional_derivatives(model, processor, all_cavs, concept, target, target_idx, target_image_path, save_path, layer_idx, num_cavs, batch_size, device, seed=1337):

    target_loader = get_dataloader(processor, target_image_path, batch_size=batch_size)
    all_activations, all_grads, all_attentions = get_activations_grads_attentions(model, target_loader, device, target_idx)

    cavs = all_cavs[concept][layer_idx] 

    # Select a random CAV out of num_cavs 
    torch.manual_seed(seed)
    rand_idx = torch.randint(low=0, high=num_cavs, size=(1,)).item()
    cav = cavs[rand_idx, :] # shape (n_embd,) 

    
    grads = all_grads[layer_idx]
    attentions = all_attentions[layer_idx]

    
    attn_directional_derivatives = get_attn_directional_derivatives(cav, grads, attentions, num_cavs=1, device=device)
    attn_directional_derivatives = attn_directional_derivatives[1:] # Remove CLS token 

    attn_directional_derivatives = attn_directional_derivatives.detach().cpu().numpy().tolist()

    data = {
        "target_image_path": os.path.basename(target_image_path).split('.')[0],
        "directional_derivatives": attn_directional_derivatives,
        "target": target,
        "concept": concept,
        "layer_idx": layer_idx,
    }

    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)

    return attn_directional_derivatives