import torch
import torch.nn.functional as F 

def set_seed(seed=1337):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True 

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_input(tokenizer, input_sentence):
    tokenizer.truncation_side = "right" 
    inputs = tokenizer(input_sentence, return_tensors="pt", truncation=True, padding=True) 
    return inputs

def get_model_output(model, inputs):
    """
    Return the model output given a tokenized input. 
    """
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs, output_attentions=True)
    
    return outputs

def get_predicted_label(outputs):
    """
    Return the predicted label from model outputs. 
    """
    logits = outputs.logits
    return logits.argmax(dim=-1)

def get_model_pred(logits, target_idx):
    """
    Return the model prediction for a given class.
    """
    outputs = F.softmax(logits, dim=1)
    return outputs[torch.arange(logits.shape[0]), target_idx]

def get_target_idx(logits):
    """
    Return the index of the predicted class. 
    """
    outputs = F.softmax(logits, dim=1)
    target_idx = torch.argmax(outputs.detach(), dim=1)
    return target_idx

def tensor_to_dict(tensor):
    """
    Convert a tensor of shape (n, 1) into a dictionary with indices as keys and tensor values as values.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.size(1) != 1:
        raise ValueError("Input must be a tensor of shape (n, 1)")
    
    values = tensor.flatten().tolist()

    result = {i: value for i, value in enumerate(values)}
    return result
