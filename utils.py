#%%

import numpy as np

#%%


def rand_norm_fixed(n, mean, sd, decimals=4):
    """Generate random numbers with fixed mean and standard deviation.

    Args:
        n (int): Number of data points
        mean (float): Mean
        sd (float): Standard deviation
        decimals (int, optional): Round numbers to the given number of decimals. Defaults to 4.

    Returns:
        list: list of numbers
    """
    x = np.random.normal(size=n)
    x = np.round((x - x.mean()) / np.std(x), decimals)
    return (mean + sd * x).tolist()


# %%


def authors():
    us = [
        "[@hauselin](https://twitter.com/hauselin)",
        "[@vicoldemburgo](https://twitter.com/vicoldemburgo)",
    ]
    np.random.shuffle(us)
    return f"{us[0]} & {us[1]}"