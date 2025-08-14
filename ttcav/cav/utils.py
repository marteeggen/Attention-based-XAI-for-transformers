import torch

def make_hooks(layer_idx, activations, grads):
    def fwd_hook(module, input, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        activations[layer_idx].append(out.squeeze(0).detach().cpu())

    def bwd_hook(module, grad_input, grad_output):
        grad = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        grads[layer_idx].append(grad.squeeze(0).detach().cpu())

    return fwd_hook, bwd_hook

def get_activations_grads_attentions(model, data_loader, device, target_idx=None):
    model.eval()
    all_activations = {}
    all_grads = {}
    all_attentions = {}

    for i, layer in enumerate(model.vit.encoder.layer):
        all_activations[i] = []
        all_grads[i] = []
        all_attentions[i] = []

    for batch_inputs, _ in data_loader:
        for i in range(batch_inputs.size(0)):
            sample_input = batch_inputs[i].unsqueeze(0).to(device)

            activations = {i: [] for i in all_activations}
            grads = {i: [] for i in all_grads}
            hooks = []

            for layer_idx, layer in enumerate(model.vit.encoder.layer):
                fwd_hook, bwd_hook = make_hooks(layer_idx, activations, grads)
                hooks.append((
                    layer.register_forward_hook(fwd_hook),
                    layer.register_full_backward_hook(bwd_hook)
                ))

            model.zero_grad()
            output = model(pixel_values=sample_input, output_attentions=True)
            logit = output.logits[0]
            attentions = output.attentions # tuple of (num_layers,) each of shape (1, num_heads, seq_len, seq_len)

            if target_idx is None:
                logit = logit[logit.argmax()]
            else:
                logit = logit[target_idx]

            logit.backward()

            for fwd, bwd in hooks:
                fwd.remove()
                bwd.remove()

            for k in activations:
                all_activations[k].append(activations[k][0])
            for k in grads:
                all_grads[k].append(grads[k][0])
            for layer_idx, attn in enumerate(attentions):
                all_attentions[layer_idx].append(attn.squeeze(0).cpu())

    for k in all_activations:
        all_activations[k] = torch.stack(all_activations[k])  
        all_grads[k] = torch.stack(all_grads[k])
        all_attentions[k] = torch.stack(all_attentions[k]) 

    return all_activations, all_grads, all_attentions


def make_activation_hook(layer_idx, activations):
    def hook(module, input, output):
        if layer_idx not in activations:
            activations[layer_idx] = []
        output = output[0] if isinstance(output, (tuple, list)) else output
        activations[layer_idx].append(output.cpu().detach())
    return hook

def get_activations(model, data_loader, device):
    activations = {}
    hooks = []

    for layer_idx, layer in enumerate(model.vit.encoder.layer):
        hook = layer.register_forward_hook(make_activation_hook(layer_idx, activations)) 
        hooks.append(hook)

    with torch.no_grad():
        for batch_inputs, _ in data_loader:
            batch_inputs = batch_inputs.to(device)  
            model(pixel_values=batch_inputs)

    for hook in hooks:
        hook.remove()

    for layer_idx in activations:
        activations[layer_idx] = torch.cat(activations[layer_idx], dim=0)

    return activations

def stack_all_activations(activations):
    stacked_activations = {}
    for layer_idx, activation_tensor in activations.items():
        batch_size, num_tokens, hidden_dim = activation_tensor.shape
        stacked_activations[layer_idx] = activation_tensor.reshape(-1, hidden_dim) 
    return stacked_activations