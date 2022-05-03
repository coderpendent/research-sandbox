import hashlib
import pandas as pd


class ExactDuplicateFinder:
    def __init__(self, data_path: str):
        """
        Uses the MD5 hash to find exact duplicates
        """
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

    def find_exact_duplicates(self, text_col: str, hash_col: str, doc_id_col: str):
        """
        Finds perfect/exact duplicates for the given text column.
        """
        self.df[hash_col] = self._hash_documents(text_col=text_col)
        self.df = self._sort_documents_by_md5_hash(hash_col=hash_col)
        self.df_unique = self._get_single_doc_per_md5_hash(hash_col=hash_col, cols_to_keep=[doc_id_col, text_col])
        doc_id_to_hash_map = dict(zip(self.df[doc_id_col], self.df[hash_col]))
        return doc_id_to_hash_map, self.df_unique
