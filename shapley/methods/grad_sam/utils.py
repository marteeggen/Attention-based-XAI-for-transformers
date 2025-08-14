import torch
import torch.nn.functional as F

def set_seed(seed=1337):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_grads(model, batch_inputs): 

    n_layers = model.config.num_hidden_layers

    batch_outputs = model(**batch_inputs, output_attentions=True)
    logits = batch_outputs.logits                
    attentions = batch_outputs.attentions  

    for att in attentions:
        att.retain_grad()

    probs = F.softmax(logits, dim=-1)
    max_probs, pred_classes = probs.max(dim=-1)  

    attention_grads = {layer_idx : [] for layer_idx in range(n_layers)} 

    for i in range(len(max_probs)):
        model.zero_grad()

        logits[i, pred_classes[i]].backward(retain_graph=True)

        for layer_idx, att in enumerate(attentions):
            grad = att.grad[i].clone()  
            attention_grads[layer_idx].append(grad)


        for att in attentions:
            if att.grad is not None:
                att.grad.zero_()
    
    for layer_idx, grads in attention_grads.items():
        attention_grads[layer_idx] = torch.stack(grads)
    
    return attention_grads


def calc_grad_sam_scores(all_attention_grads, all_attentions):
    """
    Return the scalar Grad-SAM scores for each token. 
    """
    set_seed()

    layer_vectors = []
    
    for layer_idx, grad in all_attention_grads.items():
        relu_grad = F.relu(grad.to(device)) # Shape (batch_size, n_heads, seq_len, seq_len)

        attention = all_attentions[layer_idx].clone().to(device).detach() # Shape (batch_size, n_heads, seq_len, seq_len)

        hadamard_product = attention * relu_grad

        summed_columns = hadamard_product.sum(dim=-1, keepdim=False) # Shape (batch_size, n_heads, seq_len)

        summed_heads = summed_columns.sum(dim=1, keepdim=False).squeeze() # Shape (batch_size, seq_len)

        layer_vectors.append(summed_heads)

    grad_sam_scores = sum(layer_vectors)

    return grad_sam_scores
