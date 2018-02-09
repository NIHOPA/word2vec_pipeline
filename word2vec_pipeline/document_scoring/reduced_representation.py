from sklearn.decomposition import IncrementalPCA
from utils.data_utils import load_document_vectors
from utils.os_utils import save_h5, get_h5save_object


class reduced_representation(object):

    def compute(self, method, n_components=10):

        DV = load_document_vectors(method)
        V = DV["docv"]
        clf = IncrementalPCA(n_components=n_components)

        msg = "Performing PCA on {}, ({})->({})"
        print(msg.format(method, V.shape[1], n_components))
        VX = clf.fit_transform(V)

        data = {
            "VX": VX,
            "VX_explained_variance_ratio_": clf.explained_variance_ratio_,
            "VX_components_": clf.components_,
        }

        return data

    def save(self, method, data, f_db):

        g = get_h5save_object(f_db, method)
        for key in g.keys():
            idx = g[key]["_ref"][:]

        save_h5(g[key], "VX", data["VX"][idx, :])
        save_h5(g[key], "VX_components_", data["VX_components_"])
        save_h5(g[key], "VX_explained_variance_ratio_",
                data["VX_explained_variance_ratio_"])
