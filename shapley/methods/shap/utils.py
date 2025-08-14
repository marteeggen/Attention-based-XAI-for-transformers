import shap
import numpy as np
import scipy as sp
import torch

def set_seed(seed=1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_shap_values(model, tokenizer, all_sentences):
    """
    Return SHAP values for multiple input samples.

    Implementation from https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html 
    """
    set_seed()

    def f(x): 
        tv = torch.tensor([tokenizer(v, truncation=True, max_length=tokenizer.model_max_length, padding="max_length")["input_ids"] for v in x], device=device) 
        attention_masks = torch.tensor([tokenizer(v, truncation=True, max_length=tokenizer.model_max_length, padding="max_length")["attention_mask"] for v in x], device=device)
        outputs = model(input_ids=tv, attention_mask=attention_masks)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:, 1])  
        return val

    explainer = shap.Explainer(f, tokenizer) 
    #print(type(explainer))
    shap_values = explainer(all_sentences)

    return shap_values

def trim(shap_values, context_window_size):
    """
    Remove the first and last element of the computed SHAP values corresponding to CLS and SEP tokens, respectively. If the original sequence length exceeds the model's context window, truncate the sequence from the right.
    """ 
    
    for sample in shap_values.data:
        if sample[0] != "" and sample[-1] != "":
            raise ValueError("[CLS] and/or [SEP] tokens are not included!")
    
    mod_arrs = []
    for arr in shap_values.values:
        arr = arr[1:-1] # Remove CLS and SEP tokens
        if len(arr) > context_window_size - 2: 
            arr = arr[:(context_window_size - 2)] 
        mod_arrs.append(arr)

    return mod_arrs

