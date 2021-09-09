import multiprocessing
from abc import ABC, abstractmethod
from typing import List

import gensim
import pandas as pd


class AbstractEmbedder(ABC):
    @staticmethod
    @abstractmethod
    def load_model(epochs: int = 100) -> None:
        """
        Abstract method responsible for loading a provided model.
        :param epochs: Number of iterations (epochs) the model must be trained for.
        """
        pass

    @staticmethod
    @abstractmethod
    def fetch_dataset() -> None:
        """
        Abstract method responsible for fetching a dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def string_compare(tokenised_sentence_a: str, tokenised_sentence_b: str, func) -> float:
        """
        Abstract method responsible for comparing two sentence strings.
        :param tokenised_sentence_a: First embedding to compare.
        :param tokenised_sentence_b: Second embedding to compare.
        :param func: Pass a function or a method to compare the two embeddings.
        """
        pass

    @staticmethod
    @abstractmethod
    def train_model(vector_size: int = 40, epochs: int = 50, workers: int = multiprocessing.cpu_count()) -> gensim.models.doc2vec.TaggedDocument:
        """
        Abstract method responsible for training a model from a provided dataset.
        :param vector_size: Size of the embedding vectors to be generated.
        :param epochs: Number of iterations a model must be trained for.
        :param workers:
        """
        pass

    @staticmethod
    @abstractmethod
    def embed(df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method responsible for acting as an entry point for the class.
        :param df: The DataFrame to have embeddings generated on.
        """
        pass
