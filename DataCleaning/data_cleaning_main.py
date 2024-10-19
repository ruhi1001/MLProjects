# ---------------------------------
# Created By : ruhi.ahuja
# Created Date : 19-10-2024
# ---------------------------------

"""
Here, we are learning to clean the data. We are going to follow following steps:
1. Load data, Check data, Null values
2. Filter Null values
3. Check Null values
4. Dropping
5. Replace with a Constant Value [median, average, mode, next value, previous value, constant value]
6. Used Other Columns' Info
7. Predict Missing Values Using ML Prediction
"""
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


class DataCleaning:
    def main_method(self):
        # 1. Load data, Check data, Null values:
        data = pd.read_csv("data_cleaning_input.csv")
        print(data.head())
        print(data.shape)

        # Checking how many null values we have in each dataset: (Important)
        print(data.isnull().sum())

        # Taking out percentage of null values:
        missing_percentage = (data.isnull().sum()) / (data.shape[0]) * 100
        print(np.round(missing_percentage, 2))

        # Visualizing null values:
        # sns.heatmap(data.isnull(), center=True)
        # plt.show()

        # 2. Filter Null values
        # Columns with more than 10 null values:
        more_10nulls_columns = data.columns[(data.isna().sum() > 10)]
        print(more_10nulls_columns)

        # Rows with more than 2 null values:
        print(data[data.isnull().sum(axis=1) > 2])

        # Columns with largets null values:
        print(data.isnull().sum().nlargest(3))

        # 3. Check Null values:
        print(data[data.isnull().sum(axis=1) > 0])
        # For example if someone does not married, he/she may not have dependent. So, 'nan' in dependent means no
        # dependent
        print(data[(data.Dependents.isna()) & (data.Married == "No")])
        # Let's replace NaN with 'NA' or not applicable
        data.Dependents[(data.Dependents.isna()) & (data.Married == "No")] = "NA"
        print(data.Dependents.unique())

        # 4. Drop null values:
        print(data.dropna())
        # pros: easy, fast
        # cons: loosing some data
        # if you want to change the dataset permanently:
        # data.dropna(inplace=True)

        # 5. Replace with a Constant Value:
        # based on previous value of the column:
        print(data.fillna(method="backfill"))

        # based on next value in the column:
        print(data.fillna(method="ffill"))

        # #Replace with specific value:
        index_loan_null = data.Loan_Amount_Term[data.Loan_Amount_Term.isna()].index
        print(index_loan_null)
        print(data.Loan_Amount_Term[data.Loan_Amount_Term.isna()])

        # Replace the loan amount with 360
        new_data = data.Loan_Amount_Term.fillna(value=360)
        print(new_data.iloc[index_loan_null])

        # Replace with average values:
        new_data = data.Loan_Amount_Term.fillna(value=data.Loan_Amount_Term.mean())
        print(new_data.iloc[index_loan_null])

        # Replace with mode value:
        new_data = data.Loan_Amount_Term.fillna(
            value=data.Loan_Amount_Term.mode().max()
        )
        print(new_data.iloc[index_loan_null])

        # Replace with min or max value:
        new_data = data.Loan_Amount_Term.fillna(value=data.Loan_Amount_Term.min())
        print(new_data.iloc[index_loan_null])

        # Replace with median value:
        new_data = data.Loan_Amount_Term.fillna(value=data.Loan_Amount_Term.median())
        print(new_data.iloc[index_loan_null])

        # 6. Used Other Columns' Info:
        # Lets play with "Married" column and try to replace the null values
        print(data.Married.unique())
        sns.countplot(x="Married", hue="Dependents", data=data)
        # plt.show()
        # As can be seen from the above graph, if someone does not married, it is less likely to have dependents. Hence, we can say: if married section is null value, and dependent value is zero, most probabley married column is 'No'

        print(data.Married[(data.Married.isna()) & (data.Dependents != 0)])
        data.Married[(data.Married.isna()) & (data.Dependents != 0)] = "No"
        print(data.Married.unique())

        # 7. Predict Missing Values Using ML Prediction
        # Lets predict LoanAmount missing values:
        print(
            f"we have {round(100*data.LoanAmount.isna().sum()/data.shape[0],2)} percentage of null values for loan "
            f"amount"
        )
        # To keep the original data set unchanged, I am copying the data set in another variable called "data_ML"
        data_ML = data.copy()

        # Separate the null values in LeanAmount and name it test_x and test_y: I am using columns without a null values to predict LoanAmount
        data_with_missed_loan = data_ML[data_ML.LoanAmount.isna()]
        test_y = data_with_missed_loan.LoanAmount
        test_x = data_with_missed_loan[
            [
                "Married",
                "Education",
                "ApplicantIncome",
                "CoapplicantIncome",
                "Loan_Status",
            ]
        ]
        print(test_y.shape)
        print(test_x.shape)
        print(test_x.head())

        categorical_columns = ["Married", "Education", "Loan_Status"]
        for items in categorical_columns:
            le = OneHotEncoder(drop="first")
            t = le.fit_transform(test_x[[items]]).toarray()
            test_x[items + "_binary"] = t
        test_x = test_x[
            [
                "ApplicantIncome",
                "CoapplicantIncome",
                "Married_binary",
                "Education_binary",
                "Loan_Status_binary",
            ]
        ]
        print(test_x.head())

        # Now remove the missing values of "LoanAmount" column from data_ML:
        data_ML = data_ML.iloc[data_ML.LoanAmount.dropna().index]
        print(data_ML)

        X = data_ML[
            [
                "Married",
                "Education",
                "ApplicantIncome",
                "CoapplicantIncome",
                "Loan_Status",
            ]
        ]
        y = data_ML.LoanAmount
        print(X.shape, y.shape)
        for items in categorical_columns:
            le = OneHotEncoder(drop="first")
            t = le.fit_transform(X[[items]]).toarray()
            X[items + "_binary"] = t
        X = X[
            [
                "ApplicantIncome",
                "CoapplicantIncome",
                "Married_binary",
                "Education_binary",
                "Loan_Status_binary",
            ]
        ]
        print(X.head())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.35, random_state=42
        )
        model = RandomForestRegressor(n_estimators=200)
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        print(r2_score(predict, y_test))
        missed_values = model.predict(test_x)
        print(missed_values)
        print(data.LoanAmount.isna().any())
        data.LoanAmount[data.LoanAmount.isna()] = missed_values
        print(data.LoanAmount.isna().any())


if __name__ == "__main__":
    DataCleaning().main_method()