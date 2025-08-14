import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from scipy.special import comb 
import numpy as np
from shapley.methods.grad_sam.utils import get_grads
from shapley.eval.eval import calc_faithfulness, get_masked_preds_segment, get_masked_input_ids
from shapley.utils import tokenize_input
from shapley.methods.approx_attn_shapley.utils import get_sampled_coalitions

def set_seed(seed=1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_coalition_attention_scores(attn_diag_masked_exp, diag_vals, coalitions, batch_size=16):
    B = coalitions.shape[0]
    scores = []

    for i in range(0, B, batch_size):
        attn_chunk = attn_diag_masked_exp[i:i+batch_size]  
        diag_chunk = diag_vals[i:i+batch_size]           
        coal_chunk = coalitions[i:i+batch_size]            

        attn_sym_max = torch.maximum(attn_chunk, attn_chunk.transpose(-1, -2))  

        coal_mask = coal_chunk.unsqueeze(-1) * coal_chunk.unsqueeze(-2)       

        seq_len = coal_mask.shape[-1]
        tri_mask = torch.triu(torch.ones(seq_len, seq_len, device=coal_mask.device), diagonal=1) 
        coal_mask = coal_mask * tri_mask  # Only upper triangle

        masked_sum = (attn_sym_max * coal_mask).sum(dim=(-2, -1))               

        coalition_sizes = coal_chunk.sum(dim=-1)                               
        singleton_mask = (coalition_sizes == 1)

        diag_contrib = (coal_chunk * diag_chunk).sum(dim=-1)                   

        final_score = torch.where(singleton_mask, diag_contrib, masked_sum)
        scores.append(final_score)

    return torch.cat(scores, dim=0).detach()

def approx_shapley_attn_max_mutual(model, tokenizer, sentences, labels, num_coalitions=50, sampling_strategy="kernel_shap", characteristic_value_type="attention", f1_score_perc=20, K=[0, 10, 20, 50], selected_layer_idx=[1, 3, 5, 7, 9, 11], batch_size=16):
    set_seed()

    n_heads = model.config.num_attention_heads
    n_layers = len(selected_layer_idx)
    num_repeats = n_layers*n_heads

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    
    if characteristic_value_type not in ["attention", "attention_grads"]:
        raise ValueError(f"Unknown type: {characteristic_value_type}.")

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
            for layer_idx, grad in attention_grads_dict.items():
                relu_grad = F.relu(grad.to(device)) # Shape (batch_size, n_heads, seq_len, seq_len) 
                attention = attentions[layer_idx].clone().to(device).detach() # Shape (batch_size, n_heads, seq_len, seq_len)

                hadamard_product = attention * relu_grad # Shape (batch_size, n_heads, seq_len, seq_len)
                summed_heads = hadamard_product.sum(dim=1)

                if aggregated_attn_matrix is None:
                    aggregated_attn_matrix = summed_heads
                else:
                    aggregated_attn_matrix += summed_heads
        
        aggregated_attn_matrix = aggregated_attn_matrix / (n_heads * n_layers)

        valid_token_mask = (
            (attention_masks == 1) &
            (input_ids != cls_token_id) &
            (input_ids != sep_token_id)
        ).detach() 

        mask_inv = ~valid_token_mask  # Shape (batch_size, seq_len)

        # Expand to rows and columns
        row_mask = mask_inv.unsqueeze(2)  
        col_mask = mask_inv.unsqueeze(1)  

        full_mask = row_mask | col_mask 

        aggregated_attn_matrix.masked_fill_(full_mask, float(0.0))

        B, seq_length = valid_token_mask.shape 

        num_players = valid_token_mask.sum(dim=1).detach().cpu()   
        max_num_players = torch.max(num_players)  

        approx_attn_shapley = torch.zeros(valid_token_mask.shape).to(device) # Shape (batch_size, seq_len) 
        samples = get_sampled_coalitions(num_players, seq_length, num_coalitions, num_repeats, B, sampling_strategy) # Shape (batch_size, seq_length)


        mask_expanded = valid_token_mask[:, None, :]
        coalitions = torch.where(mask_expanded == False, 0.0, samples.to(device)) 

        coalitions_with_player = coalitions.clone().float()
        coalitions_without_player = coalitions.clone().float()

        diag_values = torch.diagonal(aggregated_attn_matrix, dim1=1, dim2=2).unsqueeze(1).detach() 
        attn_diag_masked = aggregated_attn_matrix.clone().detach()
        attn_diag_masked[:, range(seq_length), range(seq_length)] = 0  
        attn_diag_masked_exp = attn_diag_masked.unsqueeze(1)

        for player in range(1, max_num_players + 1): 

            coalitions_with_player[:, :, player] = 1 
            coalitions_without_player[:, :, player] = 0 

            v_with = compute_coalition_attention_scores(attn_diag_masked_exp, diag_values, coalitions_with_player)
            v_without = compute_coalition_attention_scores(attn_diag_masked_exp, diag_values, coalitions_without_player)

            marginal_contributions = v_with - v_without

            coalition_sizes = coalitions_without_player.sum(dim=2) 
            weights = 1 / (torch.tensor(comb((num_players - 1)[:, None], coalition_sizes.cpu(), exact=False)).to(device) * num_players[:, None].to(device)) 

            weighted_contributions = marginal_contributions * weights # Shape (batch_size, num_coalitions)

            approx_attn_shapley[:, player] = weighted_contributions.sum(dim=1) 
        
            coalitions_with_player[:, :, player] = coalitions[:, :, player]
            coalitions_without_player[:, :, player] = coalitions[:, :, player]
        
        masked_approx_attn_shapley = approx_attn_shapley.masked_fill(~valid_token_mask.bool().detach(), float('-inf'))
 
        masked_input_ids_comp, masked_input_ids_suff = get_masked_input_ids(tokenizer, masked_approx_attn_shapley, input_ids, valid_token_mask, num_players, K, B) # Shape (len(K)*batch_size, seq_length)
        
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



