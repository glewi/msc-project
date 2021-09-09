import logging
from pathlib import Path

import neo4j
import pandas as pd
from neo4j import GraphDatabase

from graphtools.base_tools import BaseGraphTools


class Neo4JTools(BaseGraphTools):
    """ Contains helper methods that interface with an instance of a neo4j database. """
    driver = None # Stores the current database instance

    @classmethod
    def connect(cls, __uri, __username, __password):
        """ Method responsible for connecting to a database instance.
        :param __uri: URL location of the database instance.
        :param __username: Username for the database.
        :param __password: Password for the database.
        """
        if __username is None and __password is None:
            cls.driver = GraphDatabase.driver(__uri)
        else:
            cls.driver = GraphDatabase.driver(__uri, auth=(__username, __password))

    @classmethod
    def initalise_database(cls) -> None:
        """ Sets up the constraints for the database before ingestion.
        """
        with cls.driver.session() as session:
            result = session.write_transaction(cls._intialise_database)

    @classmethod
    def clear_database(cls) -> None:
        """
        Clears the existing database instance and prepares it for ingestion
        """
        with cls.driver.session() as session:
            result = session.write_transaction(cls._clear_database)

    @classmethod
    def add_node(cls, label: str, contents) -> None:
        """ Adds a node to the neo4j instance using a provided label and contents argument
        :param label: The label of the node to be added.
        :param contents: The contents of the node to be added.
        """
        with cls.driver.session() as session:
            result = session.write_transaction(cls._create_and_return_node, label, contents)

    @classmethod
    def add_relationship_between_nodes(cls, labelA: str, labelB: str, contentA, contentB, relationship: str) -> None:
        """ Adds a relationship between two existing neo4j nodes in a given instance by using specified node properties as arugments
        :param labelA: Label of the first node to be added.
        :param labelB: Label of the second node to be added.
        :param contentA: Content of the first node.
        :param contentB: Content of the second node.
        :param relationship: The relationship between the two nodes.
        """
        with cls.driver.session() as session:
            result = session.write_transaction(cls._create_and_return_relationship, labelA, labelB, contentA, contentB,
                                               relationship)

    @classmethod
    def populate_model(cls, df: pd.DataFrame, mock_data: bool = True) -> None:
        """
        Populates the database with values in a given DataFrame.
        :param df: The df to be ingested by the database.
        :param mock_data: Flag that tells the database to prepare for mocked data.
        """
        for row in df.itertuples(): # For each row in the dataframe.
            activity = str(row[1])
            risks = str(row[2])
            mitigations = str(row[3])

            # Create a list of the risks and mitigations
            risks = risks.split(',')
            mitigations = mitigations.split(',')

            # Add the activity to the database
            cls.add_node("Activity", str(activity))

            for risk, mitigation in zip(risks, mitigations):
                # Add and setup nodes between the risks and mitigations.
                cls.add_node("Risk", str(risk))
                cls.add_node("Mitigation", str(mitigation))
                cls.add_relationship_between_nodes("Activity", "Risk", str(activity), str(risk), "At_Risk_Of")
                cls.add_relationship_between_nodes("Risk", "Mitigation", str(risk), str(mitigation), "Is_Mitigated_By")

            if mock_data:
                company = str(row[4])
                job = str(row[5])

                # Add the mocked data and relationships to the activity node.
                cls.add_node("Company", str(company))
                cls.add_node("Job", str(job))

                cls.add_relationship_between_nodes("Job", "Company", str(job), str(company), "Performed_By")
                cls.add_relationship_between_nodes("Company", "Activity", str(company), str(activity), "Performs")
                cls.add_relationship_between_nodes("Job", "Activity", str(job), str(activity), "Associated_With")

        return

    @staticmethod
    def _intialise_database(tx):
        """ Prepare the database for initialisation
        :param tx: The transaction of the Neo4j instance.
        """
        setup_query = [
            "CREATE CONSTRAINT IF NOT EXISTS ON (n:Activity) ASSERT n.content IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS ON (n:Risk) ASSERT n.content IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS ON (n:Mitigation) ASSERT n.content IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS ON (n:Company) ASSERT n.content IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS ON (n:Job) ASSERT n.content IS UNIQUE;",
        ]

        for query in setup_query:
            result = tx.run(query)
            logging.log(logging.INFO, str(result))

        return

    @staticmethod
    def _clear_database(tx):
        """ Prepare the database for initialisation
        :param tx: The transaction of the Neo4j instance.
        """
        query = (
            "MATCH (n) DETACH DELETE n;"
        )

        result = tx.run(query)
        return result

    @staticmethod
    def _create_and_return_node(tx, label: str, content: str):
        """ Internal method that actually creates and returns a node in a given neo4j instance
        :param tx: The transaction of the Neo4j instance.
        :param label: The label of the node to be added.
        :param content: The contents of the node to be added.
        """
        query = (
            "MERGE (node:{label} {{content: $content }})"
            "RETURN node"
        ).format(label=label)  # Neo4j does not support paramaterised labels, .format is a workaround...

        result = tx.run(query, content=content)
        return result.single()[0]

    @staticmethod
    def _create_and_return_relationship(tx, labelA: str, labelB: str, contentA, contentB,
                                        relationship: str) -> neo4j.Result:
        """ Internal method that actually runs the transaction for the create relationship method.
        :param tx: The transaction of the Neo4j instance.
        :param labelA: Label of the first node.
        :param labelB: Label of the second node.
        :param contentA: Content of the first node.
        :param contentB: Content of the second node.
        :param relationship: The relationship between the two nodes.
        """
        query = (
            "MATCH (a:{labelA} {{content: $contentA}}), (b:{labelB} {{content: $contentB}})"
            "MERGE (a)-[r:{relationship}]->(b)"
            "RETURN a.content, type(r), b.content".format(labelA=labelA, labelB=labelB, relationship=relationship)
        )

        result = tx.run(query, contentA=contentA, contentB=contentB)
        return result.single()[0]


if __name__ == "__main__":
    Neo4JTools.connect("bolt://localhost:7687", None, None)
    Neo4JTools.clear_database()
    Neo4JTools.initalise_database()

    current_directory = Path(__file__).parent
    datasets_directory = current_directory.parent.joinpath("datasets")

    data = pd.read_csv(datasets_directory.joinpath("final.csv"), header=None)
    Neo4JTools.populate_model(data, True)