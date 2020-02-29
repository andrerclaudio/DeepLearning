import logging

import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Print in software terminal
logging.basicConfig(level=logging.DEBUG,
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

    predictors_train, predictors_test, validation_train, validation_test = train_test_split(predictors, answers,
                                                                                            test_size=0.25)
    classifier = Sequential()

    # Input layer
    classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    # Output layer
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    classifier.fit(predictors_train, validation_train, batch_size=10, epochs=100)

    predictions = classifier.predict(predictors_test)
    predictions = predictions > 0.5

    precision = accuracy_score(validation_test, predictions)
    matrix = confusion_matrix(validation_test, predictions)

    results = classifier.evaluate(predictors_test, validation_test)

    return
