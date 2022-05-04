import logging
import faiss
import numpy as np


class DocumentIndexer:
    def __init__(self, logger: logging.Logger, embeddings: np.array, doc_ids: list = None):
        """
        Creates a feature store using the Facebook's FAISS library to index the documents. The feature store
        is a FAISS index.  If no document IDs are provided, they are assigned sequentially from 0.

        The provided texts and optional IDs are assumed to be in the same order and correspond to the rows of
        the embedding matrix.

        The same FAISS index can be used to group documents by semantic similarity, or used as part of one of
        Haystack's many pipelines, such as question answering or keyword search.

        embeddings: Array of shape (nbr_documents, nbr_dimensions)
        doc_ids: The document IDs are used as custom IDs in the FAISS index.  If none are provided (default),
            they are sequentially assigned from 0.
        """
        if doc_ids is not None:
            # enforce ID uniqueness
            assert(len(set(doc_ids)) - len(doc_ids) == 0)
            assert(embeddings.shape[0] == len(doc_ids))
        self.logger = logger
        self.embeddings = embeddings.astype(np.float32)  # FAISS requires 32 or 16 bit floats
        self.doc_ids = doc_ids if doc_ids is not None else [i for i in range(embeddings.shape[0])]
        self.embed_dim = embeddings.shape[1]
        # placeholders
        self.faiss_index = None

    @staticmethod
    def get_invlists(index: faiss.swigfaiss.IndexIVFFlat):
        """Retrieves the FAISS IDs from an IVF FAISS index."""
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
        return np.hstack(all_ids).tolist() if len(all_ids) > 0 else []

    def _make_faiss_id_to_doc_id_map(self):
        """
        FAISS requires sequential IDs in order to re-create the embeddings from the Index.
        So this function stores a mapping of the FAISS ID in the index to the doc ID.
        """
        existing_ids = self.get_invlists(self.faiss_index)
        highest_id = max(existing_ids) + 1 if len(existing_ids) > 0 else 0
        return {i + highest_id: self.doc_ids[i] for i in range(self.embeddings.shape[0])}

    def _train_ivf_flat_faiss_index(self, nlist: int, nprobe: int):
        """
        Trains an IVF Flat index, which gains efficiency over pairwise searches by reducing the scope of the search
        via the use of Voroni cells.  This type of index trades speed to calculate the exact L2 distance.

        :param nlist: Number of Voroni cells. A rule of thumb is K * sqrt(nbr_documents), where K = some integer.
        :param nprobe: Number of neighboring Voroni cells to search.  A higher value trades speed for accuracy.
        """
        quantizer = faiss.IndexFlatL2(self.embed_dim)
        index = faiss.IndexIVFFlat(quantizer, self.embed_dim, nlist)
        index.nprobe = nprobe
        index.train(self.embeddings)
        return index

    def _train_ivf_pq_faiss_index(self, nlist: int, nprobe: int, m: int, bits: int):
        """
        Trains an IVF PQ index, which gains efficiency over pairwise searches by reducing the scope of the search
        via the use of Voroni cells, and over the IVF Flat index by approximating the L2 distance with
        Product Quantization (PQ).  PQ slices up vectors and operates on their centroids.  This type of index
        trades L2 accuracy for speed.

        :param nlist: Number of Voroni cells. A rule of thumb is K * sqrt(nbr_documents), where K = some integer.
        :param nprobe: Number of neighboring Voroni cells to search.  A higher value trades speed for accuracy.
        :param m: Number of centroid IDs in the final compressed vectors.  A lower value results in more compression.
        :param bits: Number of bits in each centroid.
        """
        quantizer = faiss.IndexFlatL2(self.embed_dim)
        index = faiss.IndexIVFPQ(quantizer, self.embed_dim, nlist, m, bits)
        index.nprobe = nprobe
        index.train(self.embeddings)
        return index

    def add_docs_to_index(self, nlist: int, nprobe: int, m: int = 8, bits: int = 8, faiss_index_type: str = 'Flat'):
        """
        Creates a FAISS index and adds the documents to it.

        :param nlist: Number of Voroni cells. A rule of thumb is K * sqrt(nbr_documents), where K = some integer.
        :param nprobe: Number of neighboring Voroni cells to search.  A higher value trades speed for accuracy.
        :param m: Number of centroid IDs in the final compressed vectors.  A lower value results in more compression.
            This is only used if faiss_index_type == 'PQ'.
        :param bits: Number of bits in each centroid.  This is only used if faiss_index_type == 'PQ'.
        :param faiss_index_type: Either 'Flat' or 'PQ'
        """
        if faiss_index_type == 'PQ':
            self.faiss_index = self._train_ivf_pq_faiss_index(nlist, nprobe, m, bits)
        else:
            self.faiss_index = self._train_ivf_flat_faiss_index(nlist, nprobe)

        # add the documents to the index
        faiss_id_to_doc_id_map = self._make_faiss_id_to_doc_id_map()
        self.faiss_index.add_with_ids(self.embeddings, np.array(list(faiss_id_to_doc_id_map)))

        assert(self.faiss_index.ntotal >= self.embeddings.shape[0])
        self.logger.info(f"{self.embeddings.shape[0]} documents added to the index.")
        self.logger.info(f"{self.faiss_index.ntotal} total documents in the index.")

        return faiss_id_to_doc_id_map

    def knn_search(self, query_vector: np.array, k: int):
        """
        Search for the k most similar documents.

        :param query_vector: Numpy doc embedding.
        :param k: Number of neighbors.
        """
        assert(self.faiss_index is not None)
        query_vector = query_vector.astype(np.float32)
        return self.faiss_index.search(query_vector, k)

    def range_search(self, query_vector: np.array, dist: float):
        """
        Searches for similar documents within the specified range.

        :param query_vector: Numpy doc embedding.
        :param dist: Distance to search for matches within.
        """
        assert(self.faiss_index is not None)
        query_vector = query_vector.astype(np.float32)
        return self.faiss_index.range_search(query_vector, dist)

    def get_embedding(self, doc_id: int):
        """
        Extracts embeddings from the FAISS index.

        :param doc_id: Determines which document's embedding to return.
        """
        assert(self.faiss_index is not None)
        self.faiss_index.make_direct_map()
        return self.faiss_index.reconstruct(doc_id)[:]

    def save_faiss_index(self, save_path: str):
        """
        Saves the specified FAISS index to a save path ending in .faiss.  This extracts the FAISS index from the
        Haystack document store.

        :param save_path: File path ending in .faiss
        """
        assert(self.faiss_index is not None)
        faiss.write_index(self.faiss_index, save_path)


# TODO: experiment to find ideal nlist and nprobe, will trade off between accuracy and speed
# TODO: in experiment ^, can use FlatL2 for ground truth
