import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from ttcav.data_processing.utils import get_all_file_paths, copy_images, sample_images, filter_dataset_ade20k
from ttcav.cav.ttcav import compute_avg_ttcav
from ttcav.cav.directional_derivatives import heatmap_attn_directional_derivatives
from ttcav.cav.train_cav import get_all_relative_cavs
import os

torch.manual_seed(1337)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
model.eval()
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads

batch_size = 16
num_cavs = 50

vit_label2id = {v.lower(): k for k, v in model.config.id2label.items()}
vit_label_map = {"zebra" : "zebra",
                 "dalmatian" : "dalmatian, coach dog, carriage dog",
                 "dog_sled" : "dogsled, dog sled, dog sleigh",
                 "fire_engine" : "fire engine, fire truck"}


target_concepts_map = {"zebra" : ["striped", "dotted", "zigzagged"],
                   "dalmatian" : ["dotted", "striped", "zigzagged"],
                   "dog_sled" : ["siberian_husky", "zebra", "corgi"],
                   "fire_engine" : ["red", "blue", "green", "yellow"]}


# concepts_broden = ["striped", "dotted", "zigzagged"]
# concepts_imagenet = ["zebra", "corgi", "siberian_husky"]

# for concept in concepts_broden:
#     source_folder = "ttcav/data/original/broden1_224/images/dtd"
#     destination_folder = f"ttcav/data/concept/{concept}"
#     copy_images(source_folder, destination_folder, concept)

# for concept in concepts_imagenet:
#     source_folder = f"ttcav/data/original/imagenet/{concept}"
#     destination_folder = f"ttcav/data/concept/{concept}"
#     sample_images(source_folder, destination_folder, num_images=120)

# for target in list(target_concepts_map.keys()):
#     source_folder = f"ttcav/data/original/imagenet/{target}"
#     destination_folder = f"ttcav/data/target/{target}"
#     sample_images(source_folder, destination_folder, num_images=200)


os.makedirs("ttcav/json_ttcav", exist_ok=True) 
os.makedirs("ttcav/json_heatmap", exist_ok=True)

target_folder_path = "ttcav/data/target" 
concept_folder_path = "ttcav/data/concept" 

layer_idx_list = [4, 5, 8, 9] 

for target, concepts in target_concepts_map.items():

    target_idx = vit_label2id.get(vit_label_map[target])

    all_cavs = get_all_relative_cavs(model, processor, concepts, concept_folder_path, n_layers, num_cavs, batch_size, device)


    save_path = f"ttcav/json_ttcav/ttcav_{target}.json"
    ttcav_scores = compute_avg_ttcav(model, processor, all_cavs, concepts, target, target_idx, target_folder_path, save_path, n_layers, num_cavs, batch_size, device)

    concept = concepts[0] # Concept assumed most related to target
    selected_target_img_folder_path = f"ttcav/data/test/target/{target}"
    target_image_path_list = get_all_file_paths(selected_target_img_folder_path)

    for layer_idx in layer_idx_list:

        for target_image_path in target_image_path_list:

            save_path = f"ttcav/json_heatmap/heatmap_{target}_concept_'{concept}'_img_'{os.path.basename(target_image_path).split('.')[0]}'_layer_'{layer_idx}'.json"
            directional_derivatives = heatmap_attn_directional_derivatives(model, processor, all_cavs, concept, target, target_idx, target_image_path, save_path, layer_idx, num_cavs, batch_size, device, seed=1337)
