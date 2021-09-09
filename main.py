from embedding.doc2vec_embedder import *
from embedding.scorers.cosine_scorer import *
from preprocessing.preprocessor import *
from tooling_manager import ToolingManager


def condense_duplicates(to_condense_df: pd.DataFrame) -> pd.DataFrame:
    """
    Condense a DataFrame by it's duplicate entries in column [0].
    All other associated are collapsed into each other.

    ==========================         ==========================
    |        (Before)        |         |         (After)        |
    | 0  |  1  |  2  |  ...  |         | 0  |  1  |  2  |  ...  |
    =========================          ==========================
    | a  |  1  | 5  |  ...   |         | a  |  1  | 5  |  ...   |
    | b  |  2  | 6  |  ...   |  ---->  | b  | 2,3 |6,7 |  ...   |   (Example)
    | b  |  3  | 7  |  ...   |         | b  |  4  | 8  |  ...   |
    | c  |  4  | 8  |  ...   |         ==========================
    ==========================
    :param to_condense_df: The DataFrame to be condensed.
    :return: The condensed DataFrame.
    """
    logging.log(logging.INFO, "Condensing Dataframe")
    condensed_df: pd.DataFrame = pd.DataFrame(
        to_condense_df.groupby(0, as_index=False)[1, 2].agg(','.join).reset_index(drop=True))  # Condense using pandas.
    return condensed_df


class Main:
    @staticmethod
    def run(raw_df: pd.DataFrame, score_threshold: float = 0.90,
            sample_size: int = 10, embed_class: AbstractEmbedder = Doc2VecEmbedder,
            score_class: AbstractScorer = CosineScorer, mock_data: bool = True,
            save_results: bool = True) -> None:
        """
        The main entrypoint of the program.
        :param raw_df: The raw dataset to be used.
        :param score_threshold: The value that the score uses to determine similarity.
        :param sample_size: Sample size of the provided dataset.
        :param embed_class: Embedding object to be used.
        :param score_class: Scoring object to be used.
        :param mock_data: Specifies whether to implement data mocking.
        """

        # Perform pre-processing on the provided dataset and perform
        # complex preprocessing on the first column in addition.
        logging.log(logging.INFO, "Pre-processing Dataset")
        processed_df = Preprocessor.preprocess(raw_df, complex_preprocess=False)
        processed_df[0] = Preprocessor.preprocess(pd.DataFrame(processed_df[0]), complex_preprocess=True)

        # Condense the results based on the output from the pre-processing stage.
        logging.log(logging.INFO, "Condensing Dataframe")
        condensed_df = pd.DataFrame(processed_df.groupby(0, as_index=False)[[1, 2]]
                                    .agg(','.join).reset_index(drop=True))  # Pretty elegant solution fixme

        # Tokenise the activities in column 0.
        token_df: pd.DataFrame = Preprocessor.tokenise(pd.DataFrame(condensed_df[0]))

        # Generate the sentence embeddings.
        logging.log(logging.INFO, "Generating Sentence Embeddings using " + str(embed_class.__class__.__name__))
        embedded_df: pd.DataFrame = embed_class.embed(pd.DataFrame(token_df))

        # Combine the tokens and related embeddings to the main dataset for easy viewing.
        for col_a, col_b in zip(token_df.columns, embedded_df.columns):
            condensed_df[col_a] = token_df[col_a]
            condensed_df[col_b] = embedded_df[col_b]

        # Score the generated sentence embeddings.
        logging.log(logging.INFO, "Scoring Sentence Embeddings using " + str(score_class.__class__.__name__))
        final_df: pd.DataFrame = score_class.score_dataframe(condensed_df, threshold=score_threshold, sample_size=sample_size)

        # Save the results if specified.
        if save_results is True:
            final_df.to_csv(datasets_directory.joinpath("final.csv"), index=False)

        # Ingest the resulting dataset to the graph database instance.
        logging.log(logging.INFO, "Ingesting Dataset (Depending on the size of the data, this may take some time)...")
        ToolingManager.ingest_data(final_df, db_credentials=(None, None), mock_data=mock_data)
        logging.log(logging.INFO, "Ingestion complete.  Check the terminal logs to connect to Neo4j and verify!")
        return


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    current_directory: Path = Path(__file__).parent
    datasets_directory: Path = current_directory.joinpath("datasets")
    models_output_directory: Path = current_directory.joinpath("models")

    raw: pd.DataFrame = pd.read_csv(datasets_directory.joinpath("raw.csv"), header=None)
    Main.run(raw_df=raw, sample_size=100, mock_data=True)
