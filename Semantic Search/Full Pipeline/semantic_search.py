import faiss
import joblib
import numpy as np
import tensorflow_hub as hub
from typing import List


class SimilaritySearcher:
    def __init__(
            self,
            faiss_index_path: str,
            faiss_id_to_doc_id_map: dict,
            embedding_model_path: str,
            post_processing_scaler_path: str
    ):
        self.index = faiss.read_index(faiss_index_path)
        self.faiss_id_to_doc_id_map = faiss_id_to_doc_id_map
        self.embed_model = hub.load(embedding_model_path)
        self.post_processing_scaler = joblib.load(post_processing_scaler_path)

    def text_similarity(self, query: List[str]):
        """
        For a given list of search terms, return all documents for each term with their distance scores.
        Results should not be limited by range search, so a very large value is used for the distance threshold.

        :param query: List of search terms
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
        indices = [int(self.faiss_id_to_doc_id_map[i]) for i in indices]

        # break up results by query
        similar_docs = [indices[start:end] for start, end in zip(limits, limits[1:])]
        similar_docs_distances = [distances[start:end] for start, end in zip(limits, limits[1:])]
        results = {}
        for q_id, q_res in enumerate(similar_docs):
            results[query[q_id]] = {res: similar_docs_distances[q_id][res_id] for res_id, res in enumerate(q_res)}
        return results
