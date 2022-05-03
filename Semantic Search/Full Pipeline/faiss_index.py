import logging
import os
import faiss
import numpy as np
from haystack.document_stores import FAISSDocumentStore


class DocumentIndexer:
    def __init__(self, logger: logging.Logger, embeddings: np.array, doc_texts: list, doc_ids: list = None):
        """
        Creates a feature store using the Facebook's FAISS library to index the documents. The feature store
        is a SQL DB that contains one or more FAISS indices.  Each index contains the document texts, embeddings,
        and [optional] document IDs.  If no document IDs are provided, they are assigned sequentially from 0.

        The provided texts and optional IDs are assumed to be in the same order and correspond to the rows of
        the embedding matrix.

        The same FAISS index can be used to group documents by semantic similarity, or used as part of one of
        Haystack's many pipelines, such as question answering or keyword search.

        embeddings: Array of shape (nbr_documents, nbr_dimensions)
        doc_texts: The document text is used by Haystack for storage in the SQL DB.
        doc_ids: The document IDs are used as custom IDs in the FAISS index.  If none are provided (default),
            they are sequentially assigned from 0.
        """
        # input checks, if no nulls in 1 of the inputs, and inputs are all the same length, all are ok
        assert(sum([e == "" for e in doc_texts]) == 0)
        if doc_ids is not None:
            assert(embeddings.shape[0] == len(doc_texts) == len(doc_ids))
            # assert(len(set(doc_ids)) - len(doc_ids) == 0)  # must be unique
        else:
            assert(embeddings.shape[0] == len(doc_texts))
        self.logger = logger
        self.embeddings = embeddings.astype(np.float32)  # FAISS requires 32 or 16 bit floats
        self.doc_texts = doc_texts
        self.doc_ids = doc_ids
        self.embed_dim = embeddings.shape[1]
        # placeholders
        self.document_store = None

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

    def add_docs_to_index(
            self,
            nlist: int,
            nprobe: int,
            m: int = 8,
            bits: int = 8,
            faiss_index_type: str = 'Flat',
            sql_url: str = "sqlite:///faiss_document_store.db",
            haystack_index_name: str = 'document',
            similarity_method: str = 'dot_product',
            duplicate_handling_method: str = "overwrite",
            save_path: str = None,
            batch_size: int = 10000,
    ):
        """
        Creates a Haystack document store and adds the documents to it.

        Haystack does not appear to support the IVFPQ type index, only the Flat, HNSW, and IVF.
        However, you can replace the IndexIVFFlat with IndexIVFPQ when you create the FAISS index
        outside of Haystack, adjust the faiss_index_factory_str argument, and it will still work.
        Haystack does not seem to notice or care.

        :param nlist: Number of Voroni cells. A rule of thumb is K * sqrt(nbr_documents), where K = some integer.
        :param nprobe: Number of neighboring Voroni cells to search.  A higher value trades speed for accuracy.
        :param m: Number of centroid IDs in the final compressed vectors.  A lower value results in more compression.
            This is only used if faiss_index_type == 'PQ'.
        :param bits: Number of bits in each centroid.  This is only used if faiss_index_type == 'PQ'.
        :param faiss_index_type: Either 'Flat' or 'PQ'
        :param sql_url: SQL DB location, defaults to a local SQLite DB file.
        :param haystack_index_name: What you want to name the index.  Haystack's doc store can save multiple indices.
        :param similarity_method: Either 'dot_product' or 'cosine'.  Haystack normalizes both to (0, 1).
        :param duplicate_handling_method: Either 'overwrite', 'skip', or 'fail'.  Determines which action should be
            taken if there are duplicate document IDs in the document store.
        :param save_path: Where to save the index, DB, or config when .save() is called.
        :param batch_size: How many documents to process in 1 batch.
        """
        if faiss_index_type == 'PQ':
            index = self._train_ivf_pq_faiss_index(nlist, nprobe, m, bits)
            faiss_index_name = f"IVF{nlist},PQ"
        else:
            index = self._train_ivf_flat_faiss_index(nlist, nprobe)
            faiss_index_name = f"IVF{nlist},Flat"

        # overwrite local SQLite if it exists
        if "sqlite" in sql_url and os.path.exists(sql_url.split("sqlite:///")[-1]):
            os.remove(sql_url.split("sqlite:///")[-1])

        self.document_store = FAISSDocumentStore(
            sql_url=sql_url,
            embedding_dim=self.embed_dim,
            faiss_index_factory_str=faiss_index_name,
            faiss_index=index,  # existing FAISS index
            return_embedding=False,  # if true, will return normalized embedding
            index=haystack_index_name,
            similarity=similarity_method,
            embedding_field="embedding",  # name of the embedding field (anything you want)
            progress_bar=True,
            duplicate_documents=duplicate_handling_method,
            faiss_index_path=save_path,
            faiss_config_path=save_path,
            isolation_level=None,  # sqlalchemy parameter for create_engine()
        )

        # put input into the format that Haystack expects
        if self.doc_ids is not None:
            haystack_input = [
                {'content': doc, 'embedding': self.embeddings[doc_id, :], 'id': self.doc_ids[doc_id]}
                for doc_id, doc in enumerate(self.doc_texts)
            ]
        else:
            haystack_input = [
                {'content': doc, 'embedding': self.embeddings[doc_id, :]}
                for doc_id, doc in enumerate(self.doc_texts)
            ]

        # add the documents to the index
        self.document_store.write_documents(
            documents=haystack_input,
            index=haystack_index_name,
            batch_size=batch_size,
            duplicate_documents=duplicate_handling_method,
        )
        self.logger.info(f"{self.document_store.faiss_indexes[haystack_index_name].ntotal} documents added to index.")

    def knn_search(self, query_vector: np.array, k: int, haystack_index_name: str = 'document'):
        """
        Search for the k most similar documents.

        :param query_vector: Numpy doc embedding.
        :param k: Number of neighbors.
        :param haystack_index_name: Identifies which index to use from the Haystack document store.
        :return:
        """
        assert(self.document_store is not None)
        assert(self.document_store.faiss_indexes[haystack_index_name] is not None)
        query_vector = query_vector.astype(np.float32)
        return self.document_store.faiss_indexes[haystack_index_name].search(query_vector, k)

    def range_search(self, query_vector: np.array, dist: float, haystack_index_name: str = 'document'):
        """
        Searches for similar documents within the specified range.

        :param query_vector: Numpy doc embedding.
        :param dist: Distance to search for matches within.
        :param haystack_index_name: Identifies which index to use from the Haystack document store.
        """
        assert(self.document_store is not None)
        assert(self.document_store.faiss_indexes[haystack_index_name] is not None)
        query_vector = query_vector.astype(np.float32)
        return self.document_store.faiss_indexes[haystack_index_name].range_search(query_vector, dist)

    def get_embedding(self, doc_id: int, haystack_index_name: str = 'document'):
        """
        Extracts embeddings from the stored FAISS index.

        :param doc_id: Determines which document's embedding to return.
        :param haystack_index_name: Identifies which index to use from the Haystack document store.
        """
        assert (self.document_store is not None)
        assert (self.document_store.faiss_indexes[haystack_index_name] is not None)
        self.document_store.faiss_indexes[haystack_index_name].make_direct_map()
        return self.document_store.faiss_indexes[haystack_index_name].reconstruct(doc_id)[:]

    def save_faiss_index(self, save_path: str, haystack_index_name: str = 'document'):
        """
        Saves the specified FAISS index to a save path ending in .faiss.  This extracts the FAISS index from the
        Haystack document store.

        :param save_path: File path ending in .faiss
        :param haystack_index_name: Identifies which index to use from the Haystack document store.
        :return:
        """
        assert (self.document_store is not None)
        assert (self.document_store.faiss_indexes[haystack_index_name] is not None)
        faiss.write_index(self.document_store.faiss_indexes[haystack_index_name], save_path)


# TODO: experiment to find ideal nlist and nprobe, will trade off between accuracy and speed
# TODO: in experiment ^, can use FlatL2 for ground truth
