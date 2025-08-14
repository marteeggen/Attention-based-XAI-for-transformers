import torch
import math
from shapley.utils import get_model_pred, get_target_idx, get_predicted_label

def set_seed(seed=1337):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_faithfulness(original_outputs, outputs_masked_comp, outputs_masked_suff, B, K=[0, 10, 20, 50]):
    """
    Compute the comprehensiveness (comp) and sufficiency (suff) metrics given original and masked outputs.
    """
    set_seed()
    original_logits = original_outputs.logits
    target_idx = get_target_idx(original_logits) 
    original_pred = get_model_pred(original_logits, target_idx) 
    masked_comp_pred = get_model_pred(outputs_masked_comp.logits, target_idx.repeat_interleave(len(K))).reshape(B, len(K)) 
    masked_suff_pred = get_model_pred(outputs_masked_suff.logits, target_idx.repeat_interleave(len(K))).reshape(B, len(K))
    
    pred_diff_comp = original_pred.unsqueeze(1) - masked_comp_pred 
    pred_diff_suff = original_pred.unsqueeze(1) - masked_suff_pred

    comp = (torch.sum(pred_diff_comp, dim=1) / (len(K) + 1)).detach().cpu().numpy() 
    suff = (torch.sum(pred_diff_suff, dim=1) / (len(K) + 1)).detach().cpu().numpy()

    return comp, suff

def get_masked_preds_segment(outputs_masked_suff, f1_score_perc=20, K=[0, 10, 20, 50]):
    """
    Return the segment of the masked outputs that gives the desired F1 score. 
    """
    set_seed()
    masked_suff_pred_label = get_predicted_label(outputs_masked_suff) 
    
    k_idx = K.index(f1_score_perc) 
    segments = masked_suff_pred_label.view(-1, len(K)) 
    segments = segments[:, k_idx].tolist() 
    return segments 

def get_masked_input_ids(tokenizer, masked_attributions, input_ids, valid_token_mask, num_tokens, K, B):
    """
    Mask the input features based on corresponding attributions. Return the masked version of the input with masking according to the comprehensiveness (comp) and sufficiency (suff) metrics. 
    """
    set_seed()
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    special_tokens = torch.tensor([cls_token_id, sep_token_id, pad_token_id]).to(device)

    masked_input_ids_comp = input_ids.clone().repeat_interleave(len(K), dim=0) 
    masked_input_ids_suff = input_ids.clone().repeat_interleave(len(K), dim=0) 
 
    idx = 0
    for batch_idx in range(B): 
    
        for k in range(len(K)):

            num_valid_tokens = valid_token_mask[batch_idx].sum().item()
            k_perc = K[k]
            num_tokens_to_mask = int(math.ceil(num_valid_tokens * (k_perc / 100))) 

            keep_mask_comp = torch.isin(masked_input_ids_comp[idx], special_tokens) 
            keep_mask_suff = torch.isin(masked_input_ids_suff[idx], special_tokens)  

            if num_tokens_to_mask == 0:
                masked_input_ids_suff[idx] = torch.where(keep_mask_suff, masked_input_ids_suff[idx], mask_token_id)
    
            else:
                top_values, top_indices = torch.topk(
                    masked_attributions[batch_idx],
                    k=num_tokens_to_mask,
                    dim=0  
                ) 

                keep_mask_suff[top_indices] = True 
                masked_input_ids_suff[idx] = torch.where(keep_mask_suff, masked_input_ids_suff[idx], mask_token_id) 

                all_indices = range(1, num_tokens[batch_idx] + 1) 
                remaining_indices = list(set(all_indices) - set(top_indices.detach().cpu().numpy())) 
                keep_mask_comp[remaining_indices] = True 
                masked_input_ids_comp[idx] = torch.where(keep_mask_comp, masked_input_ids_comp[idx], mask_token_id) 
            
            idx += 1

    return masked_input_ids_comp, masked_input_ids_suff


