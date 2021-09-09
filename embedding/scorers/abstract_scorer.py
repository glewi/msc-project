from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AbstractScorer(ABC):
    @staticmethod
    @abstractmethod
    def score_dataframe(df: pd.DataFrame, sample_size: int = 100, threshold: float = 0.90):
        """
        Abstract method that scores the similarity of a provided dataframe.
        :param df: The DataFrame to be scored.
        :param sample_size: Select a sample of the dataset for speed reasons.
        :param threshold: Scoring threshold, anything higher than this value is considered to be similar.
        """
        return

    @staticmethod
    @abstractmethod
    def score_vectors(vector_a: np.array, vector_b: np.array) -> np.array:
        """
        Abstract method that scores two individual sentence vectors.
        :param vector_a: Sentence vector A.
        :param vector_b: Sentence vector B.
        """
        return
