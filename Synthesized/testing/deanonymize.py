from sklearn.neighbors import NearestNeighbors


def match_datasets(ds_orig, ds_anon):
    """
    Finds matches between datasets and for each row from ds_anon returns index of matched row from ds_orig
    """

    nn = NearestNeighbors(1)
    nn.fit(ds_orig)

    return nn.kneighbors(ds_anon)[1]