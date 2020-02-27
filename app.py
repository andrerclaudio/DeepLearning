import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# import sys
# import tensorflow as tf
# import keras


# Print in software terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(name)s | %(levelname)s:  %(message)s',
                    datefmt='%d/%b/%Y - %H:%M:%S')

logger = logging.getLogger(__name__)


def application():
    """
    System initialization.
    :return: None
    """
    # All application has its initialization from here
    logger.info('Main application is running!')

    predictors = pd.read_csv('entradas-breast.csv')
    answers = pd.read_csv('saidas-breast.csv')

    predictors_train, predictors_test, answers_train, answers_test = train_test_split(predictors, answers,
                                                                                      test_size=0.25)
    return
