import faiss
import joblib
import numpy as np
import tensorflow_hub as hub
from typing import List


class SimilaritySearcher:
    def __init__(self, faiss_index_path: str, embedding_model_path: str, post_processing_scaler_path: str):
        self.index = faiss.read_index(faiss_index_path)
        self.embed_model = hub.load(embedding_model_path)
        self.post_processing_scaler = joblib.load(post_processing_scaler_path)

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

    def text_similarity(self, query: List[str], doc_ids: list = None):
        """
        For a given list of search terms, return all documents for each term with their distance scores.
        Results should not be limited by range search, so a very large value is used for the distance threshold.

        :param query: List of search terms
        :param doc_ids: List of the original document IDs.  The FAISS indices are sequential from 0 and they map back
            to the positional items from the original doc_ids list.
        """
        query_vector = self.post_processing_scaler.transform(
            self.embed_model(query)
        ).astype(np.float32)
        very_large_number = 1e10  # do not want to limit results, return distance to everything
        limits, distances, indices = self.index.range_search(query_vector, very_large_number)
        limits = limits.tolist()
        distances = distances.tolist()
        indices = indices.tolist()

        # map results back to their original IDs
        if doc_ids is not None:
            indices = [doc_ids[i] for i in indices]

        # break up results by query
        similar_docs = [indices[start:end] for start, end in zip(limits, limits[1:])]
        similar_docs_distances = [distances[start:end] for start, end in zip(limits, limits[1:])]
        results = {}
        for q_id, q_res in enumerate(similar_docs):
            results[query[q_id]] = {res: similar_docs_distances[q_id][res_id] for res_id, res in enumerate(q_res)}
        return results
