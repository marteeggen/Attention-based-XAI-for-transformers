import pandas as pd

def sample_test_set(tokenizer, df, text_col_name="sentence", label_col_name="label", max_len=20, num_samples_per_label=30):
    """
    Return a dataframe of num_samples randomly selected sentences that when tokenized contain less than max_seq_len tokens for each label. 
    """
    df = df.copy()
    if max_len is not None:
        df['token_count'] = df[text_col_name].apply(lambda x: len(tokenizer(x)["input_ids"]))
        df['token_count'] = df['token_count'] - 2 # Remove the [CLS] and [SEP] tokens from the total length 
        df = df[df['token_count'] <= max_len]
        df = df.drop(columns=['token_count']).reset_index(drop=True)

    sampled_dfs = []
    for label, group in df.groupby(label_col_name):
        if (num_samples_per_label != None) and (len(group) >= num_samples_per_label): # Balance dataset
            sampled_group = group.sample(n=num_samples_per_label, random_state=1337)
        else:
            sampled_group = group
        sampled_dfs.append(sampled_group)

    df = pd.concat(sampled_dfs)
    return df

def write_to_csv(df, file_name):
    df.to_csv(file_name, sep='\t', encoding='utf-8')

def read_from_csv(file_name):
    df = pd.read_csv(file_name, sep='\t', encoding='utf-8')
    return df

def get_custom_sst2_dataset(tokenizer, max_len=10, num_samples_per_label=30):
    splits = {'train': 'sst2/train-00000-of-00001.parquet', 'validation': 'sst2/validation-00000-of-00001.parquet', 'test': 'sst2/test-00000-of-00001.parquet'}
    df_test = pd.read_parquet("hf://datasets/nyu-mll/glue/" + splits["validation"])
    df_test_filtered = sample_test_set(tokenizer, df_test, text_col_name="sentence", label_col_name="label", max_len=max_len, num_samples_per_label=num_samples_per_label)
    write_to_csv(df_test_filtered, f"shapley/data/datasets/sst2_custom.csv")  

def get_custom_ag_news_dataset(tokenizer, max_len=20, num_samples_per_label=15):
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df_test = pd.read_parquet("hf://datasets/wangrongsheng/ag_news/" + splits["test"])
    df_test_filtered = sample_test_set(tokenizer, df_test, text_col_name="text", label_col_name="label", max_len=max_len, num_samples_per_label=num_samples_per_label)
    write_to_csv(df_test_filtered, f"shapley/data/datasets/ag_news_custom.csv")

def get_custom_imdb_dataset(tokenizer, num_samples_per_label=30):
    splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet', 'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
    df_test = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["test"])
    df_test_filtered = sample_test_set(tokenizer, df_test, text_col_name="text", label_col_name="label", max_len=None, num_samples_per_label=num_samples_per_label)
    write_to_csv(df_test_filtered, f"shapley/data/datasets/imdb_custom.csv")

def get_sst2_test_dataset():
    splits = {'train': 'sst2/train-00000-of-00001.parquet', 'validation': 'sst2/validation-00000-of-00001.parquet', 'test': 'sst2/test-00000-of-00001.parquet'}
    df_test = pd.read_parquet("hf://datasets/nyu-mll/glue/" + splits["validation"])
    write_to_csv(df_test, "shapley/data/datasets/sst2_test.csv")

def get_imdb_test_dataset():
    splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet', 'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
    df_test = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["test"])
    write_to_csv(df_test, "shapley/data/datasets/imdb_test.csv")

def get_ag_news_test_dataset():
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df_test = pd.read_parquet("hf://datasets/wangrongsheng/ag_news/" + splits["test"])
    write_to_csv(df_test, "shapley/data/datasets/ag_news_test.csv")  
