from scipy.stats import ttest_1samp
import numpy as np

def t_test(ttcav_scores, n_layers=12):

    p_values = {}

    for layer_idx in range(n_layers):

        scores = np.array(ttcav_scores[layer_idx])
        std_dev = np.std(scores)

        if std_dev < 1e-8:
            p_values[layer_idx] = np.nan

        else:
            t_stat, p_value = ttest_1samp(scores, popmean=0.5)

            p_values[layer_idx] = p_value  
        
    return p_values
    
