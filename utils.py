#%%

import numpy as np
import pingouin as pg
import pandas as pd

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
    x = np.random.normal(size=int(n))
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


# %%


def simulate_y(X, b, residual_sd=10):
    """Simulate pseudo-random regression response/outcome values with fixed regression coefficients.

    Args:
        X (pandas.DataFrame): pandas.DataFrame with predictors/features as columns.
        b (numpy.array): regression coefficients (e.g., np.array([b0, b1, b2, b3])).
        residual_mean (int, optional): Residual mean. Defaults to 0.
        residual_sd (int, optional): Residual standard deviation. Defaults to 10.

    Returns:
        numpy.array: simulated outcome/response values, given X and b
    """
    add0 = X.shape[0] % 2 != 0
    n = X.shape[0] // 2
    X = X.copy()
    X.loc[:, "Intercept"] = 1
    X = X[["Intercept", X.columns[:-1][0]]]
    errors1 = rand_norm_fixed(n=n, mean=0, sd=residual_sd)
    # np.random.shuffle(errors1)
    errors2 = [-i for i in errors1]  # flip errors
    # np.random.shuffle(errors2)
    if add0:
        errors1.append(0)
    errors = np.concatenate([errors1, errors2])
    y = (X @ b + errors).to_numpy()
    print(f"Requested b: {b}")
    print(
        f"Simulated b: {pg.linear_regression(X, y, add_intercept=False, coef_only=True)}"
    )
    print("Simulated outcome/response values:")
    print(y)
    return y


# %%


def simulate_x(n, min_max):
    """Simulate n pseudo-random x values for regression.

    Args:

    Returns:

    """
    addval = n % 2
    n = n // 2
    min_max[1] /= 2
    x = np.linspace(*min_max, n).round()
    x.sort()
    x = np.concatenate([x, x + min_max[1]])
    if addval:
        x = np.concatenate([[np.round(np.median(x))], x])
    return x
