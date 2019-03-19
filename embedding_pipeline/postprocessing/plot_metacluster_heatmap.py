from data_utils import load_dispersion_data
import seaborn as sns

plt = sns.plt

# Set the clustermap colormap
cmap_clustermap = sns.cubehelix_palette(
    as_cmap=True, rot=-0.3, light=1, reverse=True
)


def plot_heatmap():

    data = load_dispersion_data()
    linkage = data["linkage"]

    sns.set_context("notebook", font_scale=1.25)
    p = sns.clustermap(
        data=data["dispersion"],
        row_linkage=linkage,
        col_linkage=linkage,
        vmin=0.50,
        vmax=1.00,
        cmap=cmap_clustermap,
        figsize=(12, 10),
    )

    labels = p.data2d.columns

    # Sanity check, make sure the plotted dendrogram matches the saved values
    assert (labels == data["dendrogram_order"]).all()


if __name__ == "__main__":

    plot_heatmap()
    plt.show()
