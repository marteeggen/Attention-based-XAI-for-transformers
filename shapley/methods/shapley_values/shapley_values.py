from sklearn.metrics import f1_score
import numpy as np
import torch
import itertools
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

def convert_coalitions(binary_coalitions):
    """
    Convert binary coalitions (e.g., (0, 0, 0, 1, 1)) to player coalitions (e.g., (4, 5)). 
    """
    set_seed()
    coalitions = []
    for item in binary_coalitions:
        indices = tuple(i + 1 for i, val in enumerate(item) if val == 1) 
        coalitions.append(indices)
    return coalitions

def get_masked_sequences(tokenized_sequence, mask_token_id, binary_coalitions):
    """
    Return masked version of the original tokenized sequence in accordance with the given binary coalitions.
    """
    set_seed()

    cls_token_id = tokenized_sequence[0]
    sep_token_id = tokenized_sequence[-1]
    tokenized_sequence = tokenized_sequence[1:-1] 

    masked_sequences = []
    for coalition in binary_coalitions:
        masked_sequences.append(np.where(np.array(coalition) == 0, mask_token_id, tokenized_sequence.cpu().numpy())) 

    masked_sequences = torch.tensor(np.array(masked_sequences), dtype=torch.long, device=device)
    masked_sequences = torch.cat((torch.full((masked_sequences.shape[0], 1), cls_token_id, device=device), masked_sequences, torch.full((masked_sequences.shape[0], 1), sep_token_id, device=device)), dim=1) 
    return masked_sequences


def shapley_values(model, tokenizer, sentences, labels, f1_score_perc=20, K=[0, 10, 20, 50], batch_size=512):
    set_seed()

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    mask_token_id = tokenizer.mask_token_id

    all_inputs = tokenize_input(tokenizer, sentences)

    all_input_ids = all_inputs["input_ids"].clone().to(device)
    all_attention_masks = all_inputs["attention_mask"].clone().to(device)

    valid_token_mask = (
        (all_attention_masks == 1) &
        (all_input_ids != cls_token_id) &
        (all_input_ids != sep_token_id)
    ).to(device)

    B, _ = valid_token_mask.shape

    shapley_vals = torch.zeros(valid_token_mask.shape).to(device) 

    for sentence_idx, sentence in enumerate(sentences): 
        inputs = tokenize_input(tokenizer, sentence)
        input_ids = inputs["input_ids"].clone().squeeze().to(device)
        attention_masks = inputs["attention_mask"].clone().to(device)

        inputs = {key: val.to(device) for key, val in inputs.items()}
        original_output = model(**inputs)
        logits_softmax = torch.nn.functional.softmax(original_output.logits, dim=1)
        pred_class_idx = logits_softmax.argmax(dim=1).item()

        players = get_players(len(input_ids)-2)  
        num_players = len(players)

        binary_coalitions = list(itertools.product([0, 1], repeat=num_players))  
        coalitions = convert_coalitions(binary_coalitions)

        cf_dict = {coalition: None for coalition in coalitions} 
        attributions_dict = {player: 0 for player in players}

        for i in range(0, len(binary_coalitions), batch_size): 
            batch_coalitions = binary_coalitions[i:i + batch_size]
            
            masked_sequences = get_masked_sequences(input_ids.clone(), mask_token_id, batch_coalitions)
   
            with torch.no_grad():
                outputs = model(input_ids=masked_sequences.to(device), attention_mask=attention_masks.to(device))
            
            logits_softmax = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_class_values = logits_softmax[:, pred_class_idx]
    
            
            for coalition, max_value in zip(convert_coalitions(batch_coalitions), pred_class_values): 
                cf_dict[coalition] = max_value.item()

        for player in players:
            attributions_dict[player] = calc_shapley_value(player, players, cf_dict) 
        
        shapley_vals[sentence_idx][torch.tensor(list(attributions_dict.keys())).to(device)] = torch.tensor(list(attributions_dict.values())).to(device)        

    masked_shapley = shapley_vals.masked_fill(~valid_token_mask.bool().detach(), float('-inf'))

    num_players = valid_token_mask.sum(dim=1).detach().cpu() 
    masked_input_ids_comp, masked_input_ids_suff = get_masked_input_ids(tokenizer, masked_shapley, all_input_ids, valid_token_mask, num_players, K, B) 

    with torch.no_grad():
        outputs = model(input_ids=all_input_ids.to(device), attention_mask=all_attention_masks.to(device))
        all_attention_masks = all_attention_masks.clone().repeat_interleave(len(K), dim=0)
        outputs_masked_comp = model(input_ids=masked_input_ids_comp.to(device), attention_mask=all_attention_masks.to(device))
        outputs_masked_suff = model(input_ids=masked_input_ids_suff.to(device), attention_mask=all_attention_masks.to(device))

    comp_preds, suff_preds = calc_faithfulness(outputs, outputs_masked_comp, outputs_masked_suff, B, K)
    f1_score_preds = get_masked_preds_segment(outputs_masked_suff, f1_score_perc, K)

    f1_macro = f1_score(y_true=labels, y_pred=f1_score_preds, average="macro")
    f1_weighted = f1_score(y_true=labels, y_pred=f1_score_preds, average="weighted")
    comp = np.mean(comp_preds).item()
    suff = np.mean(suff_preds).item()

    return {
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "comp": comp,
            "suff": suff,
            "all_comp": comp_preds,
            "all_suff": suff_preds            
        }

