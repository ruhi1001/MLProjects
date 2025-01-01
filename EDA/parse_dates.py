# ---------------------------------
# Created By : ruhi.ahuja
# Created Date : 
# ---------------------------------

"""
Learning to parse dates
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class ParseDateClass:
    def main_method(self):
        landslides = pd.read_csv('catalog.csv')
        print(landslides.columns)
        print(landslides['date'].head())
        print(landslides['date'].dtype)
        print(landslides.dtypes)

        # Convert our date column to datetime type:
        landslides['date_parsed'] = pd.to_datetime(landslides['date'], format='%m/%d/%y')
        print(landslides['date_parsed'].head())

        # landslides['date_parsed'] = pd.to_datetime(landslides['Date'], infer_datetime_format=True)

        # Select the day of the month :
        day_of_month_landslides = landslides['date_parsed'].dt.day
        print(day_of_month_landslides)

        # Remove na's:
        day_of_month_landslides = day_of_month_landslides.dropna()
        sns.displot(day_of_month_landslides, kde=False, bins=31)
        plt.show()


if __name__=='__main__':
    ParseDateClass().main_method()