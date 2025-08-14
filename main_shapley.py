from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import numpy as np
import pandas as pd
import torch
import random

from shapley.data.utils import read_from_csv, get_custom_sst2_dataset, get_ag_news_test_dataset, get_imdb_test_dataset, get_sst2_test_dataset
from shapley.methods.approx_attn_shapley.approx_attn_shapley_max_mutual import approx_shapley_attn_max_mutual
from shapley.methods.attention.attention import attention
from shapley.methods.attn_shapley.attn_shapley_cls import attn_shapley_cls
from shapley.methods.attn_shapley.attn_shapley_grad_cls import attn_shapley_grad_cls
from shapley.methods.attn_shapley.attn_shapley_mutual_fast import attn_shapley_mutual_fast
from shapley.methods.attn_shapley.attn_shapley_max_mutual import attn_shapley_max_mutual
from shapley.methods.grad_sam.grad_sam_org import grad_sam
from shapley.methods.shap.shap import shap_values
from shapley.methods.shapley_values.shapley_values import shapley_values

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True 

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")



data = []
data_comp_suff = []

def timer(func, *args, **kwargs):
    label = kwargs.pop("label", None)  # Remove label if exists, default to None
    dataset = kwargs.pop("dataset", None)  # Remove dataset if exists, default to None

    start_time = time.time()  # Record the start time
    
    results = func(*args, **kwargs) 
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    print(f"{func.__name__}: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds.")
    print(f'f1_macro: {results["f1_macro"]}, f1_weighted: {results["f1_weighted"]}, comp: {results["comp"]}, suff: {results["suff"] }')
    print("\n")

    data.append({
        "time" : f"{np.round(hours, 0)}.{np.round(minutes, 0)}.{np.round(seconds, 0)}", 
        "method" : label, 
        "dataset" : dataset, 
        "f1_macro" : results["f1_macro"], 
        "f1_weighted" : results["f1_weighted"], 
        "comp" : results["comp"], 
        "suff" : results["suff"]})

    for comp in results["all_comp"]:
        data_comp_suff.append({
            "dataset": dataset,
            "method": label,
            "metric": "comp",
            "value": comp
        })

    for suff in results["all_suff"]:
        data_comp_suff.append({
            "dataset": dataset,
            "method": label,
            "metric": "suff",
            "value": suff
        })

f1_score_perc = 20
K = [0, 10, 20, 50]
batch_size = 32
selected_layer_idx= list(range(0, 12))

num_coalitions = 100

if f1_score_perc not in K:
    raise ValueError(f"Select a value in {K}.")

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2").to(device)
model.eval()
tokenizer.truncation_side = "right" 
get_custom_sst2_dataset(tokenizer, max_len=15, num_samples_per_label=None)
sst2 = read_from_csv("shapley/data/datasets/sst2_custom.csv")
sentences = sst2["sentence"].tolist()
labels = sst2["label"].tolist()

print("*** SST-2 short ***")

print("ATTENTION")
timer(attention, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2 short", label="Att")

print("SHAPLEY")
timer(shapley_values, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=512, dataset="SST-2 short", label="Shapley-Input")

print("GRAD-SAM")
timer(grad_sam, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2 short", label="Grad-SAM")

print("ATTENTION SHAPLEY")
timer(attn_shapley_grad_cls, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2 short", label="Shapley-Grad-Att-CLS")
timer(attn_shapley_cls, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2 short", label="Shapley-Att-CLS")

timer(attn_shapley_mutual_fast, model, tokenizer, sentences, labels, characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2 short", label="Shapley-Grad-Att-Mutual")
timer(attn_shapley_mutual_fast, model, tokenizer, sentences, labels, characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2 short", label="Shapley-Att-Mutual")


timer(attn_shapley_max_mutual, model, tokenizer, sentences, labels, characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2 short", label="Shapley-Grad-Att-Max-Mutual")
timer(attn_shapley_max_mutual, model, tokenizer, sentences, labels, characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2 short", label="Shapley-Att-Max-Mutual")

print("APPROX ATTENTION SHAPLEY")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="random", characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="SST-2 short", label="Approx. Shapley-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="kernel_shap", characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="SST-2 short", label="Kernel Shapley-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="random", characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="SST-2 short", label="Approx. Shapley-Grad-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="kernel_shap", characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="SST-2 short", label="Kernel Shapley-Grad-Att-Max-Mutual")

print("SHAP")
timer(shap_values, model, tokenizer, sentences, labels, abs_shap_values=True, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2 short", label="SHAP")




tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2").to(device) 
model.eval()
tokenizer.truncation_side = "right"
get_sst2_test_dataset()
sst2 = read_from_csv("shapley/data/datasets/sst2_test.csv")
sentences = sst2["sentence"].tolist()
labels = sst2["label"].tolist()

print("*** SST-2 ***")

print("ATTENTION")
timer(attention, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2", label="Att")

print("GRAD-SAM")
timer(grad_sam, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2", label="Grad-SAM")

print("ATTENTION SHAPLEY")
timer(attn_shapley_grad_cls, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2", label="Shapley-Grad-Att-CLS")
timer(attn_shapley_cls, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2", label="Shapley-Att-CLS")

timer(attn_shapley_mutual_fast, model, tokenizer, sentences, labels, characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2", label="Shapley-Grad-Att-Mutual")
timer(attn_shapley_mutual_fast, model, tokenizer, sentences, labels, characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2", label="Shapley-Att-Mutual")

print("APPROX ATTENTION SHAPLEY")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="random", characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="SST-2", label="Approx. Shapley-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="kernel_shap", characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="SST-2", label="Kernel Shapley-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="random", characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="SST-2", label="Approx. Shapley-Grad-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="kernel_shap", characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="SST-2", label="Kernel Shapley-Grad-Att-Max-Mutual")

print("SHAP")
timer(shap_values, model, tokenizer, sentences, labels, abs_shap_values=True, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="SST-2", label="SHAP")



tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news").to(device)
model.eval()
tokenizer.truncation_side = "right" 
get_ag_news_test_dataset()
ag_news = read_from_csv("shapley/data/datasets/ag_news_test.csv")
sentences = ag_news["text"].tolist()
labels = ag_news["label"].tolist()

print("*** Ag News **")

print("ATTENTION")
timer(attention, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="Ag News", label="Att")

print("GRAD-SAM")
timer(grad_sam, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="Ag News", label="Grad-SAM")

print("ATTENTION SHAPLEY")
timer(attn_shapley_grad_cls, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="Ag News", label="Shapley-Grad-Att-CLS")
timer(attn_shapley_cls, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="Ag News", label="Shapley-Att-CLS")

timer(attn_shapley_mutual_fast, model, tokenizer, sentences, labels, characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="Ag News", label="Shapley-Grad-Att-Mutual")
timer(attn_shapley_mutual_fast, model, tokenizer, sentences, labels, characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="Ag News", label="Shapley-Att-Mutual")

print("APPROX ATTENTION SHAPLEY")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="random", characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="Ag News", label="Approx. Shapley-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="kernel_shap", characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="Ag News", label="Kernel Shapley-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="random", characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="Ag News", label="Approx. Shapley-Grad-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="kernel_shap", characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="Ag News", label="Kernel Shapley-Grad-Att-Max-Mutual")

print("SHAP")
timer(shap_values, model, tokenizer, sentences, labels, abs_shap_values=True, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="Ag News", label="SHAP")



tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb").to(device)
model.eval()
tokenizer.truncation_side = "right" 
get_imdb_test_dataset()
imdb = read_from_csv("shapley/data/datasets/imdb_test.csv")
sentences = imdb["text"].tolist()
labels = imdb["label"].tolist()

print("*** IMDb ***")

print("ATTENTION")
timer(attention, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="IMDb", label="Att")

print("GRAD-SAM")
timer(grad_sam, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="IMDb", label="Grad-SAM")

print("ATTENTION SHAPLEY")
timer(attn_shapley_grad_cls, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="IMDb", label="Shapley-Grad-Att-CLS")
timer(attn_shapley_cls, model, tokenizer, sentences, labels, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="IMDb", label="Shapley-Att-CLS")

timer(attn_shapley_mutual_fast, model, tokenizer, sentences, labels, characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="IMDb", label="Shapley-Grad-Att-Mutual")
timer(attn_shapley_mutual_fast, model, tokenizer, sentences, labels, characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="IMDb", label="Shapley-Att-Mutual")

print("APPROX ATTENTION SHAPLEY")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="random", characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="IMDb", label="Approx. Shapley-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="kernel_shap", characteristic_value_type="attention", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="IMDb", label="Kernel Shapley-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="random", characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="IMDb", label="Approx. Shapley-Grad-Att-Max-Mutual")
timer(approx_shapley_attn_max_mutual, model, tokenizer, sentences, labels, num_coalitions=num_coalitions, sampling_strategy="kernel_shap", characteristic_value_type="attention_grads", f1_score_perc=f1_score_perc, K=K, selected_layer_idx=selected_layer_idx, batch_size=batch_size, dataset="IMDb", label="Kernel Shapley-Grad-Att-Max-Mutual")

print("SHAP")
timer(shap_values, model, tokenizer, sentences, labels, abs_shap_values=True, f1_score_perc=f1_score_perc, K=K, batch_size=batch_size, dataset="IMDb", label="SHAP")


df = pd.DataFrame(data)
df.to_csv("shapley_data_all.csv", index=False)

df = pd.DataFrame(data_comp_suff)
df.to_csv("shapley_data_comp_suff.csv", index=False)

