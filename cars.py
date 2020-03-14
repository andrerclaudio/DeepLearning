import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder


def cars_one_value_regression():
    base = pd.read_csv('auto.csv', encoding='ISO-8859-1')
    base = base.drop('dateCrawled', axis=1)
    base = base.drop('dateCreated', axis=1)
    base = base.drop('nrOfPictures', axis=1)
    base = base.drop('postalCode', axis=1)
    base = base.drop('lastSeen', axis=1)

    base['name'].value_counts()
    base = base.drop('name', axis=1)
    base['seller'].value_counts()
    base = base.drop('seller', axis=1)
    base['offerType'].value_counts()
    base = base.drop('offerType', axis=1)

    i1 = base.loc[base.price <= 10]
    base = base[base.price > 10]

    i2 = base.loc[base.price > 350000]
    base = base.loc[base.price < 350000]

    # base.loc[pd.isnull(base['vehicleType'])]
    # base['vehicleType'].value_counts()    # limousine

    # base.loc[pd.isnull(base['gearbox'])]
    # base['gearbox'].value_counts()  # manuell

    # base.loc[pd.isnull(base['model'])]
    # base['model'].value_counts()  # golf

    # base.loc[pd.isnull(base['fuelType'])]
    # base['fuelType'].value_counts()  # benzin

    # base.loc[pd.isnull(base['notRepairedDamage'])]
    # base['notRepairedDamage'].value_counts()  # nein

    values = {'vehicleType': 'limousine',
              'gearbox': 'manuell',
              'model': 'golf',
              'fuelType': 'benzin',
              'notRepairedDamage': 'nein'
              }

    base = base.fillna(value=values)

    predictors = base.iloc[:, 1:13].values
    real_price = base.iloc[:, 0].values

    # label_encoder_predictors = LabelEncoder()
    # predictors[:, 0] = label_encoder_predictors.fit_transform(predictors[:, 0])
    # predictors[:, 1] = label_encoder_predictors.fit_transform(predictors[:, 1])
    # predictors[:, 3] = label_encoder_predictors.fit_transform(predictors[:, 3])
    # predictors[:, 5] = label_encoder_predictors.fit_transform(predictors[:, 5])
    # predictors[:, 8] = label_encoder_predictors.fit_transform(predictors[:, 8])
    # predictors[:, 9] = label_encoder_predictors.fit_transform(predictors[:, 9])
    # predictors[:, 10] = label_encoder_predictors.fit_transform(predictors[:, 10])

    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])],
        # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        remainder='passthrough'  # Leave the rest of the columns untouched
    )
    predictors = ct.fit_transform(predictors).toarray()

    regression = Sequential()
    regression.add(Dense(units=158, activation='relu', input_dim=316))
    regression.add(Dense(units=158, activation='relu'))

    regression.add(Dense(units=1, activation='linear'))

    regression.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    regression.fit(predictors, real_price, batch_size=300, epochs=100)

    predictions = regression.predict(predictors)

    real_price.mean()
    predictions.mean()

    return None


def cars_cross_regression():
    base = pd.read_csv('auto.csv', encoding='ISO-8859-1')

    base = base.drop('dateCrawled', axis=1)
    base = base.drop('dateCreated', axis=1)
    base = base.drop('nrOfPictures', axis=1)
    base = base.drop('postalCode', axis=1)
    base = base.drop('lastSeen', axis=1)
    base = base.drop('name', axis=1)
    base = base.drop('seller', axis=1)
    base = base.drop('offerType', axis=1)

    base = base[base.price > 10]
    base = base.loc[base.price < 350000]

    values = {'vehicleType': 'limousine', 'gearbox': 'manuell',
              'model': 'golf', 'fuelType': 'benzin',
              'notRepairedDamage': 'nein'}

    base = base.fillna(value=values)

    predictors = base.iloc[:, 1:13].values
    real_price = base.iloc[:, 0].values

    # label_encoder_predictors = LabelEncoder()
    # predictors[:, 0] = label_encoder_predictors.fit_transform(predictors[:, 0])
    # predictors[:, 1] = label_encoder_predictors.fit_transform(predictors[:, 1])
    # predictors[:, 3] = label_encoder_predictors.fit_transform(predictors[:, 3])
    # predictors[:, 5] = label_encoder_predictors.fit_transform(predictors[:, 5])
    # predictors[:, 8] = label_encoder_predictors.fit_transform(predictors[:, 8])
    # predictors[:, 9] = label_encoder_predictors.fit_transform(predictors[:, 9])
    # predictors[:, 10] = label_encoder_predictors.fit_transform(predictors[:, 10])

    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])],
        # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        remainder='passthrough'  # Leave the rest of the columns untouched
    )
    predictors = ct.fit_transform(predictors).toarray()

    regression = KerasRegressor(build_fn=web_generator, epochs=100, batch_size=300)
    results = cross_val_score(estimator=regression, X=predictors, y=real_price, cv=10,
                              scoring='neg_mean_absolute_error')
    mean = results.mean()
    deviation = results.std()


def web_generator():
    regression = Sequential()
    regression.add(Dense(units=158, activation='relu', input_dim=316))
    regression.add(Dense(units=158, activation='relu'))
    regression.add(Dense(units=1, activation='linear'))
    regression.compile(loss='mean_absolute_error', optimizer='adam',
                       metrics=['mean_absolute_error'])
    return regression
