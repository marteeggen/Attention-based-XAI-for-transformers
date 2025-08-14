import torch 
import numpy as np
import os
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from ttcav.data_processing.data_loader import get_dataloader, get_dataloader_sampling_multiple_folders
from ttcav.cav.utils import get_activations, stack_all_activations

def train_cav(positive_activations, negative_activations, n_layers, seed=1337):

    X = np.concatenate([positive_activations, negative_activations], axis=0)
    y = np.concatenate([np.ones(len(positive_activations)), np.zeros(len(negative_activations))])

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=seed)
    X_train, y_train = shuffle(X, y, random_state=seed)
    classifier = LogisticRegression(random_state=seed, max_iter=1000) 
    classifier.fit(X_train, y_train)

    cav = classifier.coef_[0]
    unit_cav = cav / np.linalg.norm(cav)

    return unit_cav

def train_multiple_cavs(model, processor, positive_concept_folder, negative_concept_folders, n_layers, num_cavs, batch_size, device):

    all_cavs = {layer_idx: [] for layer_idx in range(n_layers)}

    for i in range(num_cavs):

        positive_loader = get_dataloader(processor, positive_concept_folder, batch_size=batch_size)
        negative_loader = get_dataloader_sampling_multiple_folders(processor, negative_concept_folders, num_images=len(positive_loader.dataset), batch_size=batch_size, seed=i)

        positive_activations = get_activations(model, positive_loader, device)
        negative_activations = get_activations(model, negative_loader, device)

        positive_activations = stack_all_activations(positive_activations)
        negative_activations = stack_all_activations(negative_activations)

        for layer_idx in range(n_layers):
            cav = train_cav(positive_activations[layer_idx], negative_activations[layer_idx], n_layers)
            all_cavs[layer_idx].append(torch.tensor(cav, device=device))

    stacked_cavs = {
        layer_idx: torch.stack(all_cavs[layer_idx], dim=0)  
        for layer_idx in range(n_layers)
    }

    return stacked_cavs

def get_all_relative_cavs(model, processor, concepts, concept_folder_path, n_layers, num_cavs, batch_size, device):

    all_cavs_dict = defaultdict(lambda: defaultdict(list))

    for concept in concepts:

        positive_concept_folder = os.path.join(concept_folder_path, concept) 
        negative_concept_folders = [os.path.join(concept_folder_path, c) for c in concepts if c != concept]

        all_cavs = train_multiple_cavs(model, processor, positive_concept_folder, negative_concept_folders, n_layers, num_cavs, batch_size, device)

        for layer_idx in range(n_layers):
            cavs = all_cavs[layer_idx]

            all_cavs_dict[concept][layer_idx] = cavs
    
    return all_cavs_dict



