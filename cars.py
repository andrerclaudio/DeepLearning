import pandas as pd


def autos():
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

    return None
