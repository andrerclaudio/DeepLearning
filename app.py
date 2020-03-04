import logging

import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

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
    load_neural_network()


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


def turing_adjustments():
    """

    """
    predictors = pd.read_csv('entradas-breast.csv')
    validation = pd.read_csv('saidas-breast.csv')

    classifier = KerasClassifier(build_fn=turing_web)
    parameters = {'batch_size': [10, 30],
                  'epochs': [50, 100],
                  'optimizer': ['adam', 'sgd'],
                  'loos': ['binary_crossentropy', 'hinge'],
                  'kernel_initializer': ['random_uniform', 'normal'],
                  'activation': ['relu', 'tanh'],
                  'neurons': [16, 8]}
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=5)
    grid_search = grid_search.fit(predictors, validation)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    return None


def turing_web(optimizer, loos, kernel_initializer, activation, neurons):
    """

    """
    classifier = Sequential()
    classifier.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss=loos, metrics=['binary_accuracy'])

    return classifier


def one_register_only():
    """

    """
    predictors = pd.read_csv('entradas-breast.csv')
    validation = pd.read_csv('saidas-breast.csv')

    classifier = Sequential()
    classifier.add(Dense(units=8, activation='relu', kernel_initializer='normal', input_dim=30))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=8, activation='relu', kernel_initializer='normal'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    classifier.fit(predictors, validation, batch_size=10, epochs=100)

    new = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                     0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                     0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                     0.84, 158, 0.363]])
    prediction = classifier.predict(new)
    return None


def save_neural_network():
    """

    """
    predictors = pd.read_csv('entradas-breast.csv')
    validation = pd.read_csv('saidas-breast.csv')

    classifier = Sequential()
    classifier.add(Dense(units=8, activation='relu', kernel_initializer='normal', input_dim=30))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=8, activation='relu', kernel_initializer='normal'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    classifier.fit(predictors, validation, batch_size=10, epochs=100)

    classifier_json = classifier.to_json()
    with open('classificador-breast.json', 'w') as json_file:
        json_file.write(classifier_json)
    classifier.save_weights('classificador-breast.h5')

    return None


def load_neural_network():
    """

    """
    file = open('classificador-breast.json', 'r')
    web_structure = file.read()
    file.close()

    classifier = model_from_json(web_structure)
    classifier.load_weights('classificador-breast.h5')

    new = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                     0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                     0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                     0.84, 158, 0.363]])
    prediction = classifier.predict(new)

    predictors = pd.read_csv('entradas-breast.csv')
    validation = pd.read_csv('saidas-breast.csv')
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    result = classifier.evaluate(predictors, validation)

    return None
