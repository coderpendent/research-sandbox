import logging
import pandas as pd


class DataCleaner:
    def __init__(self, data_path: str, data_colnames: list, logger: logging.Logger):
        """
        Loads and cleans data

        :param data_path: location of raw data CSV file
        :param data_colnames: names of the columns to read
        """
        self.logger = logger
        # TODO: replace pandas read from csv with read from sql, add MySQL connection details
        self.df = pd.read_csv(data_path, usecols=data_colnames)

    def _create_unique_doc_id(self, doc_id_colname: str):
        """Ensures that every row has a unique ID"""
        self.df.reset_index(drop=False, inplace=True)
        self.df.rename(columns={"index": doc_id_colname}, inplace=True)

    def _clean_colnames(self):
        self.df.columns = [col.lower().replace(" ", "_") for col in self.df.columns]

    def clean_data(self, doc_id_colname: str = "doc_id"):
        """
        Performs doc ingestion and all cleaning steps

        :param doc_id_colname: name of the unique key for every row
        """
        self._create_unique_doc_id(doc_id_colname=doc_id_colname)
        self._clean_colnames()
        return self.df
