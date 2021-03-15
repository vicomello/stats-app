#%%

import numpy as np

#%%


def rand_norm_fixed(n, mean, sd, decimals=4):
    x = np.random.normal(size=n)
    x = np.round((x - x.mean()) / np.std(x), decimals)
    return (mean + sd * x).tolist()
