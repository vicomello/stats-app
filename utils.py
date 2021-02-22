#%%

import numpy as np

#%%


def rand_norm_fixed(n, mean, sd):
    x = np.random.normal(size=n)
    x = (x - x.mean()) / np.std(x)
    return (mean + sd * x).tolist()
