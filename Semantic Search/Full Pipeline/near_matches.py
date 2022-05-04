import faiss
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class NearDuplicateFinder:
    def __init__(
            self,
            data_path: str,
            faiss_id_to_doc_id_map: dict,
            faiss_index: faiss.swigfaiss.IndexIVFFlat = None,
            faiss_index_path: str = None
    ):
        """
        Uses a FAISS index to find near duplicates
        """
        assert(faiss_index is not None or faiss_index_path is not None)
        # TODO: replace pandas read from csv with read from sql, add MySQL connection details
        self.df = pd.read_csv(data_path)
        self.faiss_id_to_doc_id_map = faiss_id_to_doc_id_map
        if faiss_index_path is not None:
            self.index = faiss.read_index(faiss_index_path)
        else:
            self.index = faiss_index

    @staticmethod
    def get_embedding(index: faiss.swigfaiss.IndexIVFFlat, faiss_id: int):
        """Retrieves the embedding for the specified FAISS ID from a FAISS index."""
        return index.reconstruct(faiss_id)[:]

    def _determine_clusters(self, dist_threshold: float = 60.):
        """
        Builds clusters from similar documents within the FAISS index.  Note that the dataset passed to this
        class (self.df) could have fewer rows than there are documents in the FAISS index, because the index
        allows documents to be added at any time.

        :param dist_threshold: The max distance to consider documents as similar.
        """
        # get embeddings from FAISS index
        self.index.make_direct_map()
        embeddings = np.vstack(
            [self.get_embedding(self.index, int(faiss_id)) for faiss_id in list(self.faiss_id_to_doc_id_map)]
        )

        # perform range search for all document embeddings
        limits, distances, indices = self.index.range_search(embeddings, dist_threshold)
        limits = limits.tolist()
        indices = indices.tolist()

        # form clusters by determining which documents are similar
        clusters = [indices[start:end] for start, end in zip(limits, limits[1:])]
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

    def find_near_duplicates(
            self,
            distance_threshold: float,
            cluster_col: str,
            doc_id_col: str,
            cols_to_keep: list,
            reduce: bool = False
    ):
        """
        Finds near duplicates for the given text column.
        """
        # get clusters of similar documents, where similarity is defined by the distance threshold
        clusters = self._determine_clusters(dist_threshold=distance_threshold)
        sparse_mat = self.build_adjacency_matrix_from_clusters(
            clusters=clusters,
            nbr_documents=len(list(self.faiss_id_to_doc_id_map))
        )
        nbr_clusters, cluster = connected_components(sparse_mat)

        # add cluster to dataframe
        original_doc_ids_from_faiss_index = [int(self.faiss_id_to_doc_id_map[i]) for i in range(len(cluster))]
        original_doc_id_to_cluster_map = dict(zip(original_doc_ids_from_faiss_index, cluster))
        self.df[cluster_col] = self.df[doc_id_col].map(original_doc_id_to_cluster_map)

        # reduce dataset to 1 example document per cluster
        cols_to_keep = [doc_id_col] + cols_to_keep
        if reduce:
            output = self.df.groupby(cluster_col)[cols_to_keep].first().reset_index(drop=True)
        else:
            output = self.df.copy()

        return nbr_clusters, output
