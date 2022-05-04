import urllib.request
import tarfile
import os
import logging
import joblib
import numpy as np
import tensorflow_hub as hub
from sklearn.preprocessing import MinMaxScaler


class USEEmbeddingModel:
    def __init__(self, logger: logging.Logger, model_path: str = None):
        """
        Universal Sentence Encoder (USE) is a state of the art semantic similarity model.
        It is preferable to BERT embeddings for semantic similarity because:
            * It was trained specifically for detecting semantic similarity with sentence pairs
            * It has a greater range of values for the embedding dimensions than BERT, allowing it
              to better separate close matches in the embedding space (0.5 - 0.8 vs 0.79 - 0.87 for BERT)
        This class pulls the pre-trained USE model from TensorFlow Hub, then uses it to
        create document level embeddings.  Note that USE has a dimensionality of 512, meaning
        only the first 512 tokens of the document will be encoded.
        """
        self.logger = logger
        self.model_path = model_path
        self.model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = None

    def _download_and_extract_tar_gz_model(self):
        url = self.model_url + "?tf-hub-format=compressed"
        fpath = "USE_Embedding_Model/"
        fname = "universal-sentence-encoder_4.tar.gz"

        # download the model to the current working directory
        urllib.request.urlretrieve(url, fname)

        # extract to a new folder
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        with tarfile.open(fname) as tar:
            tar.extractall(fpath + fname)

        # remove downloaded tar.gz file
        os.remove(fname)

    def _load_model(self):
        self.logger.info(f"Model {self.model_url} loading")
        if not os.path.isfile("USE_Embedding_Model/saved_model.pb"):
            self._download_and_extract_tar_gz_model()
        self.model = hub.load("USE_Embedding_Model")
        self.logger.info(f"Model {self.model_url} loaded")

    @staticmethod
    def batch(iterable, batch_size=1):
        """
        Creates batches of equal size, batch_size

        Example usage:
            data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # list of data

            for x in batch(data, 3):
                print(x)

            # Output

            [0, 1, 2]
            [3, 4, 5]
            [6, 7, 8]
            [9, 10]
        """
        iterable_len = len(iterable)
        for ndx in range(0, iterable_len, batch_size):
            yield iterable[ndx:min(ndx + batch_size, iterable_len)]

    def get_embeddings(self, text_input, post_processing_scaler=None, batch_size=256):
        """
        Runs text through the model and produces the embeddings

        :param text_input: a list where each item is a document
        :param post_processing_scaler: a sklearn scaler to apply to the embeddings
        :param batch_size: integer representing how many samples to include in a batch
        """
        self._load_model()
        embeddings = []
        # helper variables to track progress
        nbr_batches = int(np.ceil(len(text_input) / batch_size))
        current_batch = 1

        for batch_indices in self.batch(iterable=range(len(text_input)), batch_size=batch_size):
            progress = round(100 * current_batch / nbr_batches, 2)
            if progress % 10 == 0:
                self.logger.info(f"Embedding progress: {progress}%")

            # grab the records for this batch
            batch_records = [text_input[idx] for idx in batch_indices]

            # forward pass over the input
            model_output = self.model(batch_records)

            # save the embeddings
            embeddings.append(model_output.numpy())

            current_batch += 1

        # convert the list of embeddings to a numpy array, then scale it
        self.logger.info("Converting list to Numpy array and scaling to (0, 1) range")
        if post_processing_scaler is None:
            scaler = MinMaxScaler()
            embeddings = np.array(
                [np.array(i) for i in scaler.fit_transform(np.vstack(embeddings)).tolist()]
            )
            joblib.dump(scaler, "USE_Embedding_Model/" + "post_processing_scaler.joblib")

        return embeddings
