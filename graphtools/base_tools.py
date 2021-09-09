import neo4j
import pandas as pd


class BaseGraphTools:
    @classmethod
    def connect(cls, __uri, __username, __password) -> None:
        """ Abstract method responsible for connecting to a database instance.
        :param __uri: URL location of the database instance.
        :param __username: Username for the database.
        :param __password: Password for the database.
        """
        return

    @classmethod
    def initalise_database(cls) -> None:
        """ Sets up the constraints for the database before ingestion.
        """
        return

    @classmethod
    def clear_database(cls) -> None:
        """ Clears the existing database instance prior to ingestion
        """
        return

    @classmethod
    def add_node(cls, label: str, contents: object) -> None:
        """ Adds a node to the database instance using a provided label and contents argument
        :param label: The label of the node to be added.
        :param contents: The contents of the node to be added.
        """
        return

    @classmethod
    def add_relationship_between_nodes(cls, labelA: str, labelB: str, contentA, contentB, relationship: str) -> None:
        """ Adds a relationship between two existing  nodes in a given instance by using specified node properties as arguments
        :param labelA: Label of the first node.
        :param labelB: Label of the second node.
        :param contentA: Content of the first node.
        :param contentB: Content of the second node.
        :param relationship: The relationship between the two nodes.
        """
        return

    @classmethod
    def populate_model(cls, df: pd.DataFrame, mock_data: bool = True) -> None:
        """
        Populates the database with values in a given DataFrame.
        :param df: The df to be ingested by the database.
        :param mock_data: Flag that tells the database to prepare for mocked data.
        """
        return