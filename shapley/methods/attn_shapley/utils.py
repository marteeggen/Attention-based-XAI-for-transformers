import torch
import torch.nn.functional as F
from itertools import combinations
from shapley.methods.grad_sam.utils import get_grads
from shapley.methods.shapley_values.utils import calc_shapley_value

def set_seed(seed=1337):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_grad_attn(model, batch_inputs, attentions, layer_indices):
    """
    Return the Hadamard product of the attention matrices and the corresponding ReLU gradient of the output with respect to the attention matrices.
    """
    attention_grads_dict = get_grads(model, batch_inputs) 

    grad_attn_scores = []
    for layer_idx in layer_indices:
        grad = attention_grads_dict[layer_idx]
    
        relu_grad = F.relu(grad.to(device)) # Shape (batch_size, n_heads, seq_len, seq_len)

        attention = attentions[layer_idx].clone().to(device).detach() # Shape (batch_size, n_heads, seq_len, seq_len)

        hadamard_product = attention * relu_grad # Shape (batch_size, n_heads, seq_len, seq_len)

        grad_attn_scores.append(hadamard_product[:, :, 0, :]) 
    
    return grad_attn_scores

def get_players(num_players):
    """
    Return a list of all players starting at 1. 
    """
    return list(range(1, num_players + 1))

def create_cf_dict(characteritic_values, num_players):
    """
    Create a dictionary from a list with all possible combinations of indices as keys and the sum of the corresponding elements in the input list as values.
    """
    cf_dict = {(): 0} 

    indices = range(1, num_players + 1)  

    for r in range(1, num_players + 1):
        for combo in combinations(indices, r): 
            value = sum(characteritic_values[i - 1] for i in combo) 
            cf_dict[combo] = value

    return cf_dict

def get_attributions_dict(attn_scores, players):
    """
    Return the aggregated Shapley values for each player (token). 
    """
    set_seed()
    attributions_dict = {player: 0 for player in players}

    for attn_score in attn_scores:
        attn_score = attn_score.detach().cpu().numpy()
        cf_dict = create_cf_dict(attn_score, len(players))

        for player in players:
            attributions_dict[player] += calc_shapley_value(player, players, cf_dict) 

    return attributions_dict