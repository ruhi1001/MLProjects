# ---------------------------------
# Created By : ruhi.ahuja
# Created Date :
# ---------------------------------

"""
*** File details here
"""
import numpy as np
import pandas as pd


class HandleMissing:
    def main_method(self):
        nfl_data = pd.read_csv("NFL_data.csv")
        print(nfl_data.shape)
        missing_values_count = nfl_data.isnull().sum()
        print(missing_values_count)

        total_cells = np.product(nfl_data.shape)
        total_missing = missing_values_count.sum()

        # percent of nfl_data that is missing
        percent_missing = (total_missing / total_cells) * 100
        print(percent_missing)
        nfl_data.dropna()

        # remove all columns with at least one missing value
        columns_with_na_dropped = nfl_data.dropna(axis=1)
        columns_with_na_dropped.head()
        # just how much data did we lose?
        print("Columns in original dataset: %d \n" % nfl_data.shape[1])
        print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])

        # get a small subset of the NFL dataset
        subset_nfl_data = nfl_data.loc[:, "EPA":"Season"].head()
        print(subset_nfl_data)
        # replace all NA's with 0
        subset_nfl_data.fillna(0)

        # replace all NA's the value that comes directly after it in the same column,
        # then replace all the remaining na's with 0
        subset_nfl_data.fillna(method="bfill", axis=0).fillna(0)


if __name__ == "__main__":
    HandleMissing().main_method()