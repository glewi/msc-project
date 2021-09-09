import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from embedding.scorers.abstract_scorer import AbstractScorer


class CosineScorer(AbstractScorer):
    def __init__(self):
        CosineScorer.__name__ = "CosineScorer"

    @staticmethod
    def score_dataframe(df: pd.DataFrame, threshold: float = 0.90, sample_size: int = 100) -> pd.DataFrame:
        """
        Method that scores the similarity of a provided dataframe.
        :param df: The DataFrame to be scored.
        :param sample_size: Select a sample of the dataset for speed reasons.
        :param threshold: Scoring threshold, anything higher than this value is considered to be similar.
        """
        if sample_size is not None:
            df = df.sample(sample_size)

        embedding_list = df["Embedded Tokenised 0"].to_numpy().flatten()

        # Create the DataFrame that will store similarities.
        similarity_df = pd.DataFrame(columns=[["Index_A", "Index_B", "Similarity"]])

        # Compare every element in the list with every other element.
        for i in range(len(embedding_list)):
            for j in range(i + 1, len(embedding_list)):  # Use this to decrease redundant comparions
                score: float = CosineScorer.score_vectors(embedding_list[i], embedding_list[j])

                # Make sure the score is between acceptable levels.
                if score <= 0:
                    score = 0.0
                elif score >= 1:
                    score = 1.0

                # Generate a row that will be added to the similarity_df
                similarity_row = pd.Series([int(i), int(j), score], index=similarity_df.columns)
                similarity_df = similarity_df.append(similarity_row, ignore_index=True)

        # Filter and remove rows that are less than the specified threshold.
        similarity_df = similarity_df.replace(np.nan, 0.0)
        similarity_df = similarity_df[similarity_df >= threshold]
        similarity_df = similarity_df.dropna().reset_index(drop=True)

        # Rearrange the original and generated similarity DataFrame.
        original_df, similarities_df = CosineScorer._rearrange(df, similarity_df)

        # Merge the condensed similarity rows with the original dataframe.
        if len(similarities_df) is not 0:
            final_df = CosineScorer._merge_and_condense(original_df, similarities_df)
            logging.log(logging.INFO, "Found {0} similarities in the provided dataset ".format(len(similarities_df)))
        else:
            logging.log(logging.WARNING, "No similarities were found in the provided dataset "
                                         "(Try increasing the sample size...)")
        return final_df

    @staticmethod
    def score_vectors(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        Method that scores two individual sentence vectors.
        :param vector_a: Sentence vector A.
        :param vector_b: Sentence vector B.
        """
        if (vector_a.any() and vector_b.any()) is not (None or np.NaN):
            return 1 - cosine(vector_a, vector_b)
        else:
            return np.NaN

    @staticmethod
    def _rearrange(original_df: pd.DataFrame, similarity_df: pd.DataFrame):
        """
        Private method that uses the generated similarity index to condense and remove duplicate rows.
        @param original_df: The main DataFrame, this will have duplicate elements condensed.
        @param similarity_df: Generated DataFrame of duplicate entries with their indices and similarity scores.
        """
        logging.log(logging.INFO, "Rearranging dataframes based on similarity.")

        # Used to assemble and drop rows respectively
        rows_list = []
        drop_rows = []

        # Get all similar indices as numpy arrays for efficiency purposes.
        index_a = similarity_df["Index_A"].to_numpy().flatten()
        index_b = similarity_df["Index_B"].to_numpy().flatten()

        for a, b in zip(index_a, index_b):
            a_row = original_df.iloc[int(a)]
            b_row = original_df.iloc[int(b)]

            # Set the second duplicate row to be dropped later.
            drop_rows.append(int(b))

            # Combine the values from the two duplicate rows into the row stored at index A.
            combined: pd.Series = a_row[[1, 2]].astype(str) + ', ' + b_row[[1, 2]].astype(str)

            # Append a dict of the above info to a list that will be used to create a DataFrame later.
            rows_list.append({
                0: str(a_row[0]),
                1: str(combined[1]),
                2: str(combined[2]),
            })

        # Drop the second duplicate rows.
        original_df = original_df.drop(original_df.index[drop_rows])

        # Create a new DataFrame from the condemned similar rows.
        condensed_similar_rows_df = pd.DataFrame(rows_list)
        return original_df, condensed_similar_rows_df

    @staticmethod
    def _merge_and_condense(original_without_similar_rows_df, condensed_similar_rows_df):
        """
        Private method that combines and condenses the rows of similarities and the original DataFrame.
        :param original_without_similar_rows_df: The original DataFrame without similar rows included.
        :param condensed_similar_rows_df: The condensed rows of similar rows.
        :return:
        """
        condensed_similar_rows_df = condensed_similar_rows_df[[0, 1, 2]]
        original_without_similar_rows_df = original_without_similar_rows_df[[0, 1, 2]]

        # Append and remove
        ndf = pd.concat([condensed_similar_rows_df, original_without_similar_rows_df], ignore_index=True)
        ndf = ndf.groupby(0, as_index=False)[1, 2].agg(','.join).reset_index(drop=True)
        return ndf

