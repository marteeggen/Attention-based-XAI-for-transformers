import torch
import numpy as np
from sklearn.metrics import f1_score
from shapley.utils import tokenize_input
from shapley.eval.eval import calc_faithfulness, get_masked_preds_segment, get_masked_input_ids
from shapley.methods.grad_sam.utils import get_grads, calc_grad_sam_scores

def set_seed(seed=1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def grad_sam(model, tokenizer, sentences, labels, f1_score_perc=20, K=[0, 10, 20, 50], batch_size=16):
    set_seed()
    n_heads = model.config.num_attention_heads 
    n_layers = model.config.num_hidden_layers
    
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

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
        attention_grads = get_grads(model, batch_inputs) 

        attention_masks = batch_inputs["attention_mask"].clone().detach() 
        input_ids = batch_inputs["input_ids"].clone().detach()

        B, _ = input_ids.shape

        seq_lengths = attention_masks.sum(dim=1) 
        num_tokens = seq_lengths - 2 

        grad_sam_scores = calc_grad_sam_scores(attention_grads, attentions)
        grad_sam_scores = grad_sam_scores / (n_layers*n_heads*seq_lengths.unsqueeze(-1))

        valid_token_mask = (
            (attention_masks == 1) &
            (input_ids != cls_token_id) &
            (input_ids != sep_token_id)
        ).to(device) 
        
        masked_grad_sam_scores = grad_sam_scores.masked_fill(~valid_token_mask.bool().detach(), float('-inf'))

        masked_input_ids_comp, masked_input_ids_suff = get_masked_input_ids(tokenizer, masked_grad_sam_scores, input_ids, valid_token_mask, num_tokens, K, B)

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





