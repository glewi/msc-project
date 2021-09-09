import logging
import random
from typing import Tuple, List

import neo4j
import numpy as np
import pandas as pd

from graphtools.neo4j_tools import Neo4JTools


class ToolingManager:
    @staticmethod
    def ingest_data(ingestion_df: pd.DataFrame, db_credentials: Tuple[str, str], mock_data: bool = True) -> None:
        """
        Ingest the dataset to the Neo4j instance.
        :param ingestion_df: DataFrame to be ingested.
        :param db_credentials: Database username/password, default to None.
        :param mock_data: Specify if the data should have mocking added.
        """
        try:
            try:
                Neo4JTools.connect(
                    "bolt://graph-db:7687", db_credentials[0], db_credentials[1])
            except:
                Neo4JTools.connect(
                    "bolt://localhost:7687", db_credentials[0], db_credentials[1])
        except neo4j.exceptions.ServiceUnavailable as e:
            logging.log(logging.ERROR, str(e))

        Neo4JTools.initalise_database()
        Neo4JTools.clear_database()

        if mock_data:
            ingestion_df: pd.DataFrame = ToolingManager.add_mock_data(ingestion_df)

        Neo4JTools.populate_model(ingestion_df, mock_data=mock_data)
        return

    @staticmethod
    def add_mock_data(to_mock_df: pd.DataFrame) -> pd.DataFrame:
        """
         Add mocked data to the dataset before graph database ingestion.
        :param to_mock_df: DataFrame to add mocking.
        :return: DataFrame with adding mocking information.
        """
        mock_companies = ["Company_A", "Company_B", "Company_C"]

        # Generate a random list of job ID numbers, and a random list of the above mock companies.
        jobs: List[int] = list(np.random.randint(low=1, high=10, size=to_mock_df.shape[0]))
        companies: List[str] = [random.choice(mock_companies) for i in range(to_mock_df.shape[0])]

        # Append these to the DataFrame
        to_mock_df["Company"] = companies
        to_mock_df["Job"] = jobs
        return to_mock_df
