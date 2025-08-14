import os
from collections import defaultdict
import json
from ttcav.data_processing.utils import convert
from ttcav.data_processing.data_loader import get_dataloader
from ttcav.cav.directional_derivatives import get_attn_directional_derivatives
from ttcav.cav.utils import get_activations_grads_attentions


def compute_ttcav(cavs, grads, attentions, num_cavs, device): 
    """Input: CAVs and gradients associated with a specific layer."""

    attn_directional_derivatives = get_attn_directional_derivatives(cavs, grads, attentions, num_cavs, device) # shape (batch, n_tokens, num_cavs)

    # Aggregate directional derivatives 
    attn_directional_derivatives = attn_directional_derivatives.mean(dim=1) # shape (batch, num_cavs) 
    
    positive_counts = (attn_directional_derivatives > 0).sum(dim=0)
    ttcav_scores = positive_counts.float() / attn_directional_derivatives.size(0) # shape: (num_cavs,) 

    return ttcav_scores.detach().cpu().numpy()


def compute_avg_ttcav(model, processor, all_cavs, concepts, target, target_idx, target_folder_path, save_path, n_layers, num_cavs, batch_size, device): 

    target_folder_path = os.path.join(target_folder_path, target) 
    target_loader = get_dataloader(processor, target_folder_path, batch_size=batch_size)
    all_activations, all_grads, all_attentions = get_activations_grads_attentions(model, target_loader, device, target_idx)

    all_ttcav_scores = defaultdict(dict) 

    for concept in concepts:

        for layer_idx in range(n_layers):
            cavs = all_cavs[concept][layer_idx]
            grads = all_grads[layer_idx]
            attentions = all_attentions[layer_idx]

            ttcav_scores = compute_ttcav(cavs, grads, attentions, num_cavs, device) 
            all_ttcav_scores[concept][layer_idx] = ttcav_scores

    with open(save_path, 'w') as f:
        json.dump(convert(all_ttcav_scores), f, indent=4)
    
    return all_ttcav_scores





