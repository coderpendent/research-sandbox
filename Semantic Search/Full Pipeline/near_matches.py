import faiss
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class NearDuplicateFinder:
    def __init__(self, data_path: str, faiss_index: faiss.swigfaiss.IndexIVFFlat = None, faiss_index_path: str = None):
        """
        Uses a FAISS index to find near duplicates
        """
        assert(faiss_index is not None or faiss_index_path is not None)
        # TODO: replace pandas read from csv with read from sql, add MySQL connection details
        self.df = pd.read_csv(data_path)
        if faiss_index_path is not None:
            self.index = faiss.read_index(faiss_index_path)
        else:
            self.index = faiss_index

    @staticmethod
    def get_invlists(index: faiss.swigfaiss.IndexIVFFlat):
        """Retrieves the document indices from an IVF FAISS index."""
        index = faiss.extract_index_ivf(index)
        invlists = index.invlists
        all_ids = []
        for list_nbr in range(index.nlist):
            ls = invlists.list_size(list_nbr)
            if ls == 0:
                continue
            all_ids.append(
                faiss.rev_swig_ptr(invlists.get_ids(list_nbr), ls).copy()
            )
        return np.hstack(all_ids).tolist()

    @staticmethod
    def get_embedding(index: faiss.swigfaiss.IndexIVFFlat, doc_id: int):
        """Retrieves the embedding for the specified doc_id from a FAISS index."""
        return index.reconstruct(doc_id)[:]

    def _determine_clusters(self, dist_threshold: float = 60.):
        """
        Builds clusters from similar documents.

        :param dist_threshold: The max distance to consider documents as similar.
        """
        # get embeddings from FAISS index
        self.index.make_direct_map()
        embeddings = np.vstack([self.get_embedding(self.index, doc_id) for doc_id in self.get_invlists(self.index)])

        # perform range search for all document embeddings
        limits, distances, doc_ids = self.index.range_search(embeddings, dist_threshold)

        # form clusters by determining which documents are similar
        doc_ids = doc_ids.tolist()
        limits = limits.tolist()
        clusters = [doc_ids[start:end] for start, end in zip(limits, limits[1:])]
        return {i: set(v) for i, v in enumerate(clusters)}

    @staticmethod
    def build_adjacency_matrix_from_clusters(clusters: dict, nbr_documents: int):
        """
        Given the clusters, build a square, sparse adjanceny matrix that shows which documents
        are connected (documents are connected if they are in the same cluster).

        :param clusters: The cluster dictionary from determine_clusters()
        :param nbr_documents: The total number of documents.  The matrix will have this many rows and columns.
        """
        # create a list of tuples representing the edges in a graph
        edges = set([(x, y) for c in clusters.values() for x, y in combinations_with_replacement(c, 2)])

        # create a square adjacency matrix from the edge list
        matrix_shape = (nbr_documents, nbr_documents)
        rows, cols = zip(*edges)
        return coo_matrix((np.ones(len(edges)), (rows, cols)), shape=matrix_shape)

    def find_near_duplicates(self, distance_threshold: float, cluster_col: str, cols_to_keep: list):
        """
        Finds near duplicates for the given text column.
        """
        # get clusters of similar documents, where similarity is defined by the distance threshold
        clusters = self._determine_clusters(dist_threshold=distance_threshold)
        sparse_mat = self.build_adjacency_matrix_from_clusters(
            clusters=clusters,
            nbr_documents=len(self.get_invlists(self.index))
        )
        nbr_clusters, cluster = connected_components(sparse_mat)

        # reduce dataset to 1 example document per cluster
        # TODO: do not rely on row order, change this to a dict mapping on search_hit_id
        self.df[cluster_col] = cluster
        return nbr_clusters, self.df.groupby(cluster_col)[cols_to_keep].first().reset_index(drop=True)
