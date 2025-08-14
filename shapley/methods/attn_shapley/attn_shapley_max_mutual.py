from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
import numpy as np
import itertools
from shapley.methods.grad_sam.utils import get_grads
from shapley.eval.eval import calc_faithfulness, get_masked_preds_segment, get_masked_input_ids
from shapley.utils import tokenize_input
from shapley.methods.attn_shapley.utils import get_players
from shapley.methods.shapley_values.utils import calc_shapley_value 

def set_seed(seed=1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_subset_attention_scores(attn_matrix, num_players):
    """
    Computes attention scores for all subsets of tokens.
    """

    characteristic_values = {}

    tokens = list(range(num_players))

    for r in range(num_players + 1): 
        for subset in itertools.combinations(tokens, r):

            subset_1idx = tuple(i + 1 for i in subset)

            if len(subset) == 0: 
                characteristic_values[subset_1idx] = 0.0

            elif len(subset) == 1:  
                i = subset[0]
                characteristic_values[subset_1idx] = float(attn_matrix[i, i])

            else:
                score = 0.0
                for i, j in itertools.combinations(subset, 2): 
                    score += max(float(attn_matrix[i, j]), float(attn_matrix[j, i]))

                characteristic_values[subset_1idx] = score

    return characteristic_values

def attn_shapley_max_mutual(model, tokenizer, sentences, labels, characteristic_value_type="attention", f1_score_perc=20, K=[0, 10, 20, 50], batch_size=16):
    set_seed()

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    n_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers

    inputs = tokenize_input(tokenizer, sentences)
        
    total_samples = len(sentences) 

    f1_score_preds = []
    comp_preds = []
    suff_preds = []

    for start_idx in range(0, total_samples, batch_size):

        end_idx = min(start_idx + batch_size, total_samples)
        batch_inputs = {key: value[start_idx:end_idx].to(device) for key, value in inputs.items()}

        batch_outputs = model(**batch_inputs, output_attentions=True)

        attentions = batch_outputs.attentions

        attention_masks = batch_inputs["attention_mask"].clone().detach()
        input_ids = batch_inputs["input_ids"].clone().detach()

        if characteristic_value_type == "attention":
            aggregated_attn_matrix = None
            for attention in attentions:
                summed_heads = attention.sum(dim=1)
                if aggregated_attn_matrix is None:
                    aggregated_attn_matrix = summed_heads
                else:
                    aggregated_attn_matrix += summed_heads

        elif characteristic_value_type == "attention_grads": 
            attention_grads_dict = get_grads(model, batch_inputs) 

            aggregated_attn_matrix = None
            for layer_idx in range(n_layers):
                grad = attention_grads_dict[layer_idx]
                relu_grad = F.relu(grad.to(device)) # Shape (batch_size, n_heads, seq_len, seq_len)
                attention = attentions[layer_idx].clone().to(device).detach() # Shape (batch_size, n_heads, seq_len, seq_len)

                hadamard_product = attention * relu_grad # Shape (batch_size, n_heads, seq_len, seq_len)
                summed_heads = hadamard_product.sum(dim=1)

                if aggregated_attn_matrix is None:
                    aggregated_attn_matrix = summed_heads
                else:
                    aggregated_attn_matrix += summed_heads

        else:
            raise ValueError(f"Unknown type: {characteristic_value_type}.")

        valid_token_mask = (
            (attention_masks == 1) &
            (input_ids != cls_token_id) &
            (input_ids != sep_token_id)
        ).to(device)

        B, _ = valid_token_mask.shape

        attn_shapley = torch.zeros(valid_token_mask.shape).to(device)

        seq_lengths = attention_masks.sum(dim=1).to(device) 
        num_players = valid_token_mask.sum(dim=1).detach().cpu() 

        for batch_idx, attn_matrix in enumerate(aggregated_attn_matrix):
            start = 1  
            end = 1 + num_players[batch_idx]  
            attn_clean = attn_matrix[start:end, start:end] / (n_heads * n_layers)
            cf_dict = compute_subset_attention_scores(attn_clean, num_players[batch_idx])
            players = get_players(seq_lengths[batch_idx]-2)

            for player in players:
                attn_shapley[batch_idx][player] = torch.tensor(calc_shapley_value(player, players, cf_dict), dtype=attn_shapley.dtype).to(device)


        masked_attn_shapley = attn_shapley.masked_fill(~valid_token_mask.bool().detach(), float('-inf'))

        masked_input_ids_comp, masked_input_ids_suff = get_masked_input_ids(tokenizer, masked_attn_shapley, input_ids, valid_token_mask, num_players, K, B)
        
        with torch.no_grad():
            attention_masks = attention_masks.clone().repeat_interleave(len(K), dim=0) 
            outputs_masked_comp = model(input_ids=masked_input_ids_comp.to(device), attention_mask=attention_masks.to(device))
            outputs_masked_suff = model(input_ids=masked_input_ids_suff.to(device), attention_mask=attention_masks.to(device))
        
        comp, suff = calc_faithfulness(batch_outputs, outputs_masked_comp, outputs_masked_suff, B, K)
        comp_preds.append(comp)
        suff_preds.append(suff)

        f1_score_preds.extend(get_masked_preds_segment(outputs_masked_suff, f1_score_perc, K))

        torch.cuda.empty_cache()

    f1_macro = f1_score(y_true=labels, y_pred=f1_score_preds, average="macro")
    f1_weighted = f1_score(y_true=labels, y_pred=f1_score_preds, average="weighted")
    comp = np.mean(np.concatenate(comp_preds)).item()
    suff = np.mean(np.concatenate(suff_preds)).item()

    return {
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "comp": comp,
            "suff": suff,
            "all_comp": np.concatenate(comp_preds),
            "all_suff": np.concatenate(suff_preds)
        }