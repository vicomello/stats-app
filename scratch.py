#%%
import altair as alt
from vega_datasets import data

source = data.movies.url

#%%
stripplot = (
    alt.Chart(source, width=40)
    .mark_circle(size=8)
    .encode(
        x=alt.X(
            "jitter:Q",
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        y=alt.Y("IMDB_Rating:Q"),
        color=alt.Color("Major_Genre:N", legend=None),
        column=alt.Column(
            "Major_Genre:N",
            header=alt.Header(
                labelAngle=-90,
                titleOrient="top",
                labelOrient="bottom",
                labelAlign="right",
                labelPadding=3,
            ),
        ),
    )
    .transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
    )
    .configure_facet(spacing=0)
    .configure_view(stroke=None)
)
#%%
stripplot
# %%


fig5 = (
    alt.Chart(df_all)
    .transform_density(
        density="Happiness",
        as_=["Happiness", "density"],
        bandwidth=2.0,
        groupby=["Species"],
    )
    .mark_area(orient="horizontal", opacity=0.3, color="#f98e09")
    .encode(
        alt.X(
            "density:Q",
            title="",
            stack="zero",
            impute=None,
            axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
        ),
        alt.Y(
            "Happiness:Q",
        ),
        color="Species:N",
    )
    .properties(height=fig_height)
)


# %%

import altair as alt
from vega_datasets import data

alt.Chart(data.cars()).transform_density(
    "Miles_per_Gallon",
    as_=["Miles_per_Gallon", "density"],
    extent=[5, 50],
    groupby=["Origin"],
).mark_area(orient="horizontal").encode(
    y="Miles_per_Gallon:Q",
    color="Origin:N",
    x=alt.X(
        "density:Q",
        stack="center",
        impute=None,
        title=None,
        axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
    ),
    column=alt.Column(
        "Origin:N",
        header=alt.Header(
            titleOrient="bottom",
            labelOrient="bottom",
            labelPadding=0,
        ),
    ),
).properties(
    width=100
).configure_facet(
    spacing=0
).configure_view(
    stroke=None
)


# %%

fig_violin = (
    alt.Chart(df_all)
    .transform_density(
        density="Happiness",
        as_=["Happiness", "density"],
        bandwidth=2.0,
        groupby=["Species"],
    )
    .mark_area(orient="horizontal", opacity=0.3, color="#f98e09")
    .encode(
        alt.X(
            "density:Q",
            title="",
            stack="center",
            impute=None,
            axis=alt.Axis(grid=False, ticks=True),
        ),
        alt.Y(
            "Happiness:Q",
        ),
        color=alt.Color("Species"),
        column=alt.Column(
            "Species:N",
            header=alt.Header(
                titleOrient="bottom",
                labelOrient="bottom",
                labelPadding=0,
            ),
        ),
    )
    .properties(height=fig_height, width=200)
    .configure_facet(spacing=0)
    .configure_view(stroke=None)
)


# %%

fig_violin = (
    alt.Chart(df_all)
    .transform_density("Happiness", as_=["Happiness", "density"], groupby=["Species"])
    .mark_area(orient="horizontal")
    .encode(
        y="Happiness:Q",
        color="Species",
        x=alt.X(
            "density:Q",
            stack="zero",
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, grid=False, ticks=True, values=[-0.1, 0, 0.1]),
        ),
    )
    .properties(width=100)
)


# %%

b0 = 3
b1 = 5
pd.DataFrame({"b0": [b0], "b1": [b1], "Model": f"y = {b0} + {b1} * x"})

b0
df["happiness"] = utils.rand_norm_fixed(n=df.shape[0], mean=b0 + df["Age"] * b1, sd=5)
pg.linear_regression(df[["Age"]], df["happiness"])


b0, b1 = 17.5, 0.87
errors1 = utils.rand_norm_fixed(n=int(df.shape[0] / 2), mean=0, sd=5)
errors2 = [-i for i in errors1]
eps = np.concatenate([errors1, errors2])
eps.mean()
eps.std()
df["happiness"] = b0 + df["Age"] * b1 + eps
pg.linear_regression(df[["Age"]], df["happiness"])


n = 9
n // 2
df[["Age", "i"]] @ np.array([17.5, 0.87])

df.apply(lambda x: len(x.unique()))
df["abc"] = "abc"
df["aaa"] = 1.0
df["aba"] = 1


def simulate_regression(X, b, residual_mean=0, residual_sd=10, add_intercept=True):
    if X.shape[0] % 2 != 0:
        add0 = True
    else:
        add0 = False
    n = X.shape[0] // 2
    X.loc[:, "Intercept"] = 1
    errors1 = utils.rand_norm_fixed(n=n, mean=0, sd=residual_sd)
    errors2 = [-i for i in errors1]  # flip errors
    if add0:
        errors2.append(0)
    errors = np.concatenate([errors1, errors2])
    y = X @ b + errors
    print(
        f"Simulated b: {pg.linear_regression(X, y, add_intercept=False, coef_only=True)}"
    )

    X = X.drop(columns="Intercept")
    X.loc[:, "y"] = y
    return X


simulate_regression(df[["Age"]], np.array([17.4, 3.2]))

# %%
