import logging
import time
from pathlib import Path
from typing import List

import contractions
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Preprocessor:
    @staticmethod
    def preprocess(original_df: pd.DataFrame, complex_preprocess: bool = False) -> pd.DataFrame:
        """
        Pre-processor entrypoint.  Branches off to different parts of the class.
        :param original_df: The DataFrame to be processed.
        :param complex_preprocess: Specify whether to use complex pre-processing or not.
        :return: Pre-processed DataFrame.
        """
        nltk.download('stopwords')
        nltk.download('punkt')
        stop_words = set(stopwords.words('english'))

        if complex_preprocess:
            logging.log(logging.INFO, "Complex Dataframe Preprocessing")
            return Preprocessor._preprocess_dataframe(original_df)
        else:
            logging.log(logging.INFO, "Simple Dataframe Preprocessing")
            return Preprocessor._simple_preprocess_dataframe(original_df)

    @staticmethod
    def tokenise(original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tokenise a DataFrame.
        :param original_df: The DataFrame to be tokenised.
        :return: The tokenised DataFrame.
        """
        token_df: pd.DataFrame = original_df.copy()

        for i in token_df.columns:
            logging.log(logging.INFO, "Tokenising Column: {0}".format(str(i)))
            new_col: str = "Tokenised {0}".format(str(i))

            # Append the new tokenised column.
            token_df[new_col]: pd.DataFrame = token_df[i].apply(lambda row_value: Preprocessor._tokenise_string(row_value))
            token_df = token_df.drop(token_df.columns[i], axis=1)

        return token_df

    @staticmethod
    def _simple_preprocess_dataframe(original_df: pd.DataFrame) -> pd.DataFrame:  # Process a dataframe of strings
        """
        Perform a simple pre-process of the dataframe.
        :param original_df: DataFrame to be utilised.
        :return: Pre-processed DataFrame.
        """
        for i in original_df.columns:
            logging.log(logging.INFO, "Processing Column: {0}".format(str(i)))
            original_df[i] = original_df[i].apply(lambda x: Preprocessor._simple_preprocess_string(x))
        return original_df

    @staticmethod
    def _preprocess_dataframe(original_df: pd.DataFrame) -> pd.DataFrame:  # Process a dataframe of strings
        """
        Perform a complex pre-process of the dataframe.
        :param original_df: DataFrame to be utilised.
        :return: Pre-processed DataFrame.
        """
        for i in original_df.columns:
            logging.log(logging.INFO, "Processing Column: {0}".format(str(i)))
            original_df[i] = original_df[i].apply(lambda x: Preprocessor._complex_preprocess_string(x))
        return original_df

    @staticmethod
    def _tokenise_string(sentence: str) -> pd.DataFrame:
        """
        Token a given sentence string.
        :param sentence: The sentence to be tokenised.
        :return: The tokenised sentence.
        """
        tokens: List[str] = word_tokenize(sentence)

        # Remove stopwords from the sentence.
        result: List[str] = filter(
            lambda token: token not in stopwords.words('english'), tokens)
        result: List[str] = filter(
            lambda element: element is not stopwords.words('english'), result)

        # Convert the word list to lowercase and remove excessive whitespace.
        result: List[str] = list([word.lower().strip() for word in result])

        return result

    @staticmethod
    def _simple_preprocess_string(sentence: str) -> str:  # Process a string.
        """
        Perform a simple pre-processing pass on a given sentence.
        :param sentence: Sentence string to be processed.
        :return: The pre-processed string.
        """
        # There's got to be a better way of doing this, but it works for now. TODO
        sentence = sentence.replace("-", "")
        sentence = sentence.replace("|", "")
        sentence = sentence.replace(",", "")
        sentence = sentence.replace(".", "")
        sentence = sentence.replace("!", "")
        sentence = sentence.replace("%", "")
        sentence = sentence.replace("?", "")
        sentence = sentence.replace("&", "and")
        sentence = sentence.replace("/", "and")
        sentence = sentence.replace("(", "")
        sentence = sentence.replace(")", "")
        sentence = sentence.replace("[", "")
        sentence = sentence.replace("]", "")

        # These shows up in the dataset, not sure what caused it; unicode or ascii issues?
        sentence = sentence.replace("â€™", "'")
        sentence = sentence.replace("â€˜", "")
        sentence = sentence.replace("â€”", "")

        result = sentence.lower().strip()
        return result

    @staticmethod
    def _complex_preprocess_string(sentence: str) -> str:  # Process a string.
        """
        Perform a complex pre-process on a string.
        :param sentence: Sentence to be pre-processed.
        :return: The new sentence.
        """
        sentence = Preprocessor._simple_preprocess_string(sentence)
        sentence = contractions.fix(sentence)
        return sentence


if __name__ == "__main__":
    current_directory = Path(__file__).parent
    datasets_directory = current_directory.parent.parent.joinpath("datasets")

    # Download stopwords and punctuation tools from the NLTK repository, this is only performed once.
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

    df = pd.DataFrame(pd.read_csv(
        datasets_directory.joinpath("condensed.csv"), header=None))
    print("Pre-processing Dataset...")
    t0 = time.process_time()
    df = Preprocessor.simple_preprocess_dataframe(df)
    print("Preprocessing Time: " + str(time.process_time() - t0))
    df.to_csv(datasets_directory.joinpath(
        "simple-processed.csv"), header=False, index=False)
