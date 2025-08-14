import torch
import numpy as np
from shapley.utils import tokenize_input
from shapley.methods.shap.utils import get_shap_values, trim
from shapley.eval.eval import calc_faithfulness, get_masked_preds_segment, get_masked_input_ids
from sklearn.metrics import f1_score

def set_seed(seed=1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def shap_values(model, tokenizer, sentences, labels, abs_shap_values=True, f1_score_perc=20, K=[0, 10, 20, 50], batch_size=16):
    set_seed()
    
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    inputs = tokenize_input(tokenizer, sentences)
 
    total_samples = len(sentences) 

    f1_score_preds = []
    comp_preds = []
    suff_preds = []

    for start_idx in range(0, total_samples, batch_size):
   
        end_idx = min(start_idx + batch_size, total_samples)
        batch_sentences = sentences[start_idx:end_idx]
        
        batch_inputs = {key: value[start_idx:end_idx].to(device) for key, value in inputs.items()}
        attention_masks = batch_inputs["attention_mask"].clone().detach()
        input_ids = batch_inputs["input_ids"].clone().detach()

        batch_outputs = model(**batch_inputs)


        shap_vals = get_shap_values(model, tokenizer, batch_sentences)
        shap_vals = trim(shap_vals, context_window_size=tokenizer.model_max_length) # Trim if SHAP values are given for tokens outside the context window 
        if abs_shap_values:
            shap_vals = [np.abs(arr) for arr in shap_vals] 
            

        valid_token_mask = (
            (attention_masks == 1) &
            (input_ids != cls_token_id) &
            (input_ids != sep_token_id)
        ).to(device)
        B, _ = valid_token_mask.shape
        num_players = valid_token_mask.sum(dim=1).detach().cpu()

        shap = torch.zeros(valid_token_mask.shape).to(device) 
        for i, attributions in enumerate(shap_vals):
            indices = torch.arange(1, len(attributions) + 1)
            shap[i, indices] = torch.tensor(attributions, dtype=torch.float, device=device) 
        
        masked_shap = shap.masked_fill(~valid_token_mask.bool().detach(), float('-inf'))
        
        masked_input_ids_comp, masked_input_ids_suff = get_masked_input_ids(tokenizer, masked_shap, input_ids, valid_token_mask, num_players, K, B)
        
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





