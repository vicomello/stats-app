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