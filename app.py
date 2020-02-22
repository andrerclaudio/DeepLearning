import logging


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
    