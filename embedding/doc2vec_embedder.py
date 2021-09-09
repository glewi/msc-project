import ast
import logging
import multiprocessing
import os
from pathlib import Path
from typing import List

import gensim
import gensim.downloader as api
import pandas as pd

from embedding.abstract_embedder import AbstractEmbedder

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
dirname = os.path.dirname(__file__)


class Doc2VecEmbedder(AbstractEmbedder):
    model = None
    dataset = api.load("text8")
    data = [data for data in dataset]

    @classmethod
    def load_model(cls, epochs: int = 100) -> None:
        """
        Method responsible for loading a provided model.
        :param epochs: Used to select the right file name.
        """
        models_directory: Path = Path(__file__).parent.joinpath("models")
        model: Path = models_directory.joinpath("d2v-e{0}.model".format(epochs))
        cls.model = gensim.models.doc2vec.Doc2Vec.load(str(model))

    @classmethod
    def fetch_dataset(cls) -> None:
        """
        Method responsible for fetching a dataset.
        """
        dataset = api.load("text8")

    @classmethod
    def string_compare(cls, tokenised_sentence_a: str, tokenised_sentence_b: str, func) -> float:
        """
        Compare two sentence embeddings.
        :param tokenised_sentence_a: First embedding.
        :param tokenised_sentence_b: Second embedding.
        :param func: Comparison function to be used.
        """
        vec1 = cls.model.infer_vector(tokenised_sentence_a)
        vec2 = cls.model.infer_vector(tokenised_sentence_b)
        return func(vec1, vec2)

    @classmethod
    def train_model(cls, vector_size: int = 40, epochs: int = 50, workers: int = multiprocessing.cpu_count()):
        """
        Method responsible for training a model from a provided dataset.
        :param vector_size: Size of the embedding vectors to be generated.
        :param epochs: Number of iterations a model must be trained for.
        :param workers:
        """
        data_for_training = list(Doc2VecEmbedder._tag_documents(cls.data))

        cls.model = gensim.models.doc2vec.Doc2Vec(vector_size, epochs, workers)
        cls.model.build_vocab(data_for_training)
        cls.model.train(
            data_for_training, total_examples=cls.model.corpus_count, epochs=cls.model.epochs)
        cls.model.save(
            dirname + f'/Models/d2v-custom-e{cls.model.epochs}.model')

    @staticmethod
    def embed(df: pd.DataFrame):
        """
        Abstract method responsible for acting as an entry point for the class.
        :param df: The DataFrame to have embeddings generated on.
        """
        if Doc2VecEmbedder.model is None:
            try:
                Doc2VecEmbedder.load_model()
            except FileNotFoundError as e:
                logging.log(logging.ERROR, e)

        return Doc2VecEmbedder._generate_embeddings(df)

    @classmethod
    def _generate_embeddings(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Private method used for generating embeddings on a given DataFrame.
        :param df: DataFrame to be utilised
        """
        for i in df.columns:
            logging.log(logging.INFO, "Embedding Column: {0}".format(str(i)))
            new_col: str = "Embedded {0}".format(str(i))
            df[new_col]: pd.DataFrame = df[i].apply(lambda x: cls.model.infer_vector(ast.literal_eval(str(x))))
        return df

    @classmethod
    def _tag_documents(cls, document_list: List[List[str]]) -> gensim.models.doc2vec.TaggedDocument:
        """
        A private, Doc2Vec specific method that creates a tagged document object that gensim required.
        :param document_list: List of documents fetched from the specified corpus.
        """
        for i, list_of_words in enumerate(document_list):
            yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
