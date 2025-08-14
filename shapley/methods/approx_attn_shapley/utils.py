import torch
from scipy.special import comb 
import numpy as np

def set_seed(seed=1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def shap_kernel_weight(z_size, M):
    """
    Compute the KernelSHAP weight for a coalition of size z_size (https://arxiv.org/pdf/2410.04883).
    """
    set_seed()
    if z_size == 0 or z_size == M:
        return 10**6 

    c = comb(M, z_size, exact=False)
    if c == 0:
        c = 10**(-15) # Avoid division by zero
    return (M - 1) / (c * z_size * (M - z_size))

def kernel_shap_sampling(num_players, seq_length, num_coalitions, num_repeats, batch_size):
    """
    Sample coalitions according to the KernelSHAP weight formula.
    """
    set_seed()
    all_probabilities = [] 
    for batch_idx in range(batch_size):
        M = num_players[batch_idx]

        # Create probabilities for all coalition sizes 
        weights = np.array([shap_kernel_weight(z_size, M) for z_size in range(M + 1)])
        probabilities = weights / np.sum(weights)
        probabilities = np.clip(probabilities, 0, 1) 
        all_probabilities.append(probabilities)

    # Generate binary vectors for sampled coalitions
    samples = sample_kernel_shap_coalitions(all_probabilities, num_players, seq_length, num_coalitions, num_repeats, batch_size)
    return samples


def sample_kernel_shap_coalitions(probabilities, num_players, seq_length, num_coalitions, num_repeats, batch_size):
    """
    Sample coalitions based on the weight distribution. The coalitions sizes are found based on the given probabilities, whereas the features to be included in these coalition are chosen at random.
    """
    set_seed()

    coalition_sizes = []
    for n, p in zip(num_players, probabilities):
        p_trimmed = p[1:-1] 
        p_trimmed /= p_trimmed.sum()
        sampled_sizes = np.random.choice(range(1, n), size=num_coalitions - 2, p=p_trimmed, replace=True) 
        coalition_sizes.append(np.concatenate(([0], sampled_sizes, [n]))) 
    coalition_sizes = torch.tensor(np.array(coalition_sizes)).to(device)

    coalitions = torch.zeros((batch_size, num_coalitions, seq_length), dtype=torch.int).to(device)
    for batch_idx in range(batch_size):
        used_coalitions = set()  
        
        for coalition_idx in range(num_coalitions):  
            coalition_size = coalition_sizes[batch_idx, coalition_idx].item()
            
            if coalition_size > 0:
                num_players_batch = num_players[batch_idx]
                
                retry_limit = 10  
                for _ in range(retry_limit):
                    rand_indices = torch.randperm(num_players_batch, device=device)[:coalition_size] + 1 
                    subset_tuple = tuple(sorted(rand_indices.tolist()))  

                    if subset_tuple not in used_coalitions:
                        used_coalitions.add(subset_tuple)
                        break  
                
                # Set selected features to 1
                coalitions[batch_idx, coalition_idx].scatter_(-1, rand_indices, torch.ones(num_players_batch, dtype=torch.int, device=device))
        
    return coalitions 

def sample_random_coalitions(seq_length, num_coalitions, num_repeats, batch_size):
    """
    Sample random coalitions.
    """
    set_seed()

    samples = torch.randint(0, 2, (batch_size, num_coalitions, seq_length), dtype=torch.float32) 

    return samples

def get_sampled_coalitions(num_players, seq_length, num_coalitions, num_repeats, batch_size, sampling_strategy="random"):
    set_seed()

    if sampling_strategy == "random":
        coalitions = sample_random_coalitions(seq_length, num_coalitions, num_repeats, batch_size)

    elif sampling_strategy == "kernel_shap":
        coalitions = kernel_shap_sampling(num_players, seq_length, num_coalitions, num_repeats, batch_size)

    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}. Valid sampling strategies: 'random' or 'kernel_shap'.")
    
    return coalitions


