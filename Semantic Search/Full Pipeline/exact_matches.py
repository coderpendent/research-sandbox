import logging
import hashlib
import pandas as pd


class ExactDuplicateFinder:
    def __init__(self, logger: logging.Logger, data_path: str):
        """
        Uses the MD5 hash to find exact duplicates

        :param data_path: location of data CSV file
        """
        self.logger = logger
        # TODO: replace pandas read from csv with read from sql, add MySQL connection details
        self.df = pd.read_csv(data_path)
        self.df_unique = None

    @staticmethod
    def md5(data: str, hex_representation: bool = True):
        """
        Applies MD5 hash to the data and returns the hex_representation (default)
        or the byte representation.
        """
        hasher = hashlib.md5(data.encode('utf-8'))
        return hasher.hexdigest() if hex_representation else hasher.digest()

    def _hash_documents(self, text_col: str):
        return self.df[text_col].apply(self.md5)

    def _sort_documents_by_md5_hash(self, hash_col: str):
        return self.df.sort_values(hash_col)

    def _get_single_doc_per_md5_hash(self, hash_col: str, cols_to_keep: tuple):
        """
        Given a Pandas dataframe, a hash column name, and columns to return,
        collapse the dataframe to 1 row per hash.  The rows will have duplicate
        hashes but different values for cols_to_keep, however the hashed data
        will all be the same, so it does not matter which record is returned.
        The default here is the first one.
        """
        return self.df.groupby(hash_col)[cols_to_keep].first().reset_index(drop=True)

    def find_exact_duplicates(self, text_col: str, hash_col: str, doc_id_col: str, cols_to_keep: list = []):
        """
        Finds perfect/exact duplicates for the given text column.

        :param text_col: the column with the document text
        :param hash_col: the name of the column to be created with the document text's hash
        :param doc_id_col: the column with a unique row ID (document ID)
        :param cols_to_keep: other columns to be retained in the output
        """
        # ensure that the doc_id_col is unique, if not, coerce it and warn the user
        if len(self.df) - len(set(self.df[doc_id_col])) != 0:
            self.logger.warning((
                f"The provided document ID column, {doc_id_col}, does not have unique values. "
                f"Only the first record for each document ID will be retained. Duplicates will be dropped."
            ))
            self.df = self.df.groupby(doc_id_col).first().reset_index(drop=True)

        # map every doc_id to its text's MD5 hash
        self.df[hash_col] = self._hash_documents(text_col=text_col)
        self.df = self._sort_documents_by_md5_hash(hash_col=hash_col)
        doc_id_to_hash_map = dict(zip(self.df[doc_id_col], self.df[hash_col]))

        # reduce dataset to 1 row per MD5 hash
        cols_to_keep = [doc_id_col, text_col] + cols_to_keep
        self.df_unique = self._get_single_doc_per_md5_hash(hash_col=hash_col, cols_to_keep=cols_to_keep)

        return doc_id_to_hash_map, self.df_unique
