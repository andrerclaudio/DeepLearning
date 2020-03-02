import logging

import keras
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# Print in software terminal
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s | %(name)s | %(levelname)s:  %(message)s',
                    datefmt='%d/%b/%Y - %H:%M:%S')

logger = logging.getLogger(__name__)


def application():
    """
    """
    # All application has its initialization from here
    logger.info('Main application is running!')
    cross_breast_cancer()


def cross_breast_cancer():
    """

    """
    predictors = pd.read_csv('entradas-breast.csv')
    validation = pd.read_csv('saidas-breast.csv')

    classifier = KerasClassifier(build_fn=generate_web, epochs=100, batch_size=10)
    results = cross_val_score(estimator=classifier, X=predictors, y=validation, cv=10, scoring='accuracy')
    mean = results.mean()
    deviation = results.std()
    return None


def generate_web():
    """

    """
    classifier = Sequential()
    classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return classifier


def simple_breast_cancer():
    """

    """
    predictors = pd.read_csv('entradas-breast.csv')
    answers = pd.read_csv('saidas-breast.csv')

    predictors_train, predictors_test, validation_train, validation_test = train_test_split(predictors, answers,
                                                                                            test_size=0.25)
    classifier = Sequential()

    # Input layer
    classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))

    # Output layer
    classifier.add(Dense(units=1, activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    classifier.fit(predictors_train, validation_train, batch_size=10, epochs=100)

    weight_0 = classifier.layers[0].get_weights()
    weight_1 = classifier.layers[1].get_weights()
    weight_2 = classifier.layers[2].get_weights()
    weight_3 = classifier.layers[3].get_weights()

    predictions = classifier.predict(predictors_test)
    predictions = predictions > 0.5

    precision = accuracy_score(validation_test, predictions)
    matrix = confusion_matrix(validation_test, predictions)

    results = classifier.evaluate(predictors_test, validation_test)

    return
