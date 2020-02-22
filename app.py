import logging
import sys
import tensorflow as tf
import keras


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
    logger.info('Python: %s', sys.version)
    logger.info('TensorFlow: %s', tf.__version__)
    logger.info('Keras: %s', keras.__version__)