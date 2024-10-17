# ---------------------------------
# Created By : ruhi.ahuja
# Created Date :
# ---------------------------------

"""
We are predicting the flight delays
"""
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


class MyClass:
    def main_method(self):
        # fetching data into csv
        flights = pd.read_csv('flights.csv')
        print(flights.head())
        print(flights.shape)
        flights_needed_data = flights[0:100000]
        print(flights_needed_data.shape)
        print(flights_needed_data.info())
        # print(flights_needed_data.value_counts('DIVERTED')) # 224 flights were diverted

        sb.jointplot(data=flights_needed_data, x='SCHEDULED_ARRIVAL', y='ARRIVAL_TIME')
        numneric_data = flights_needed_data.select_dtypes(include=[np.number])
        corr = numneric_data.corr(method='pearson')
        sb.heatmap(corr)
        print(corr)
        plt.show()

        # Data Preprocessing:
        flights_needed_data = flights_needed_data.drop(['YEAR', 'FLIGHT_NUMBER', 'AIRLINE', 'DISTANCE', 'TAIL_NUMBER',
                                                        'TAXI_OUT',
                                                        'SCHEDULED_TIME', 'DEPARTURE_TIME', 'WHEELS_OFF',
                                                        'ELAPSED_TIME',
                                                        'AIR_TIME', 'WHEELS_ON', 'DAY_OF_WEEK', 'TAXI_IN',
                                                        'CANCELLATION_REASON', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                                                        'ARRIVAL_TIME'
                                                        ], axis=1)

        flights_needed_data = flights_needed_data.fillna(flights_needed_data.mean())
        result = []
        for row in flights_needed_data['ARRIVAL_DELAY']:
            if row > 15:
                result.append(1)
            else:
                result.append(0)

        flights_needed_data['result'] = result
        print(flights_needed_data['result'].value_counts())
        flights_needed_data = flights_needed_data.drop(['ARRIVAL_DELAY'], axis=1)
        # Splitting data for training and test:
        data = flights_needed_data.values
        X, y = data[:, :-1], data[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        scaled_features = StandardScaler().fit_transform(X_train, X_test)
        print(scaled_features)

        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        pred_prob = clf.predict_proba(X_test)
        print(pred_prob)
        auc_score = roc_auc_score(y_test, pred_prob[:, 1])
        print(auc_score)


if __name__ == '__main__':
    MyClass().main_method()