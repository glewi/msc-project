import logging
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

# Setup logging functions.
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class ParallelTool:
    """
    Parallel tools class responsible for splitting a dataframe apply across threads.
    THIS WAS UNUSED IN THE FINAL IMPLEMENTATION.
    """


    @staticmethod
    def parallel_process(df, func):
        # After Parallel Processing
        p = Pool(cpu_count())  # Data parallelism Object
        df2 = df.copy()
        cols = df.columns.values.tolist()
        df2 = p.map(func, df[cols])

        return df2

    @staticmethod
    def parallel_apply(data, func, num_of_processes=cpu_count()):
        """ TODO


        @data:
        @func:
        @num_of_processes:
        """
        data_split = np.array_split(data, num_of_processes)
        thread_pool = Pool(num_of_processes)

        # TODO
        logging.log(
            logging.INFO, "Processing {} columns using threadpool.".format(
                len(data)))

        data = pd.concat(thread_pool.map(func, data_split))
        thread_pool.close()
        thread_pool.join()
        return data

    @staticmethod
    def run_on_subset(func, data_subset):
        return data_subset.apply(func)

    @staticmethod
    def parallelise_on_rows(data, func, num_of_processes=cpu_count()):
        """ TODO """
        return parallelise(data, partial(run_on_subset, func), num_of_processes)


def parallelise(data, func, num_of_processes=cpu_count()):  # TODO Tweak the data argument
    """ TODO """
    data_split = np.array_split(data, num_of_processes)
    thread_pool = Pool(num_of_processes)

    # TODO
    logging.log(
        logging.INFO, "Processing {} columns using threadpool.".format(
            len(data_split)))

    data = pd.concat(thread_pool.map(func, data_split))
    thread_pool.close()
    thread_pool.join()
    return data


def run_on_subset(func, data_subset):
    """ TODO """
    return data_subset.apply(func)


def parallelise_on_rows(data, func, num_of_processes=cpu_count()):
    """ TODO """
    return parallelise(data, partial(run_on_subset, func), num_of_processes)
