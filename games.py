import pandas as pd


def multiples_output_regression():
    """

    """
    base = pd.read_csv('games.csv')

    base = base.drop('Other_Sales', axis=1)
    base = base.drop('Global_Sales', axis=1)
    base = base.drop('Developer', axis=1)
