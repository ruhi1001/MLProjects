# ---------------------------------
# Created By : ruhi.ahuja
# Created Date : 
# ---------------------------------

"""
*** File details here
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class MyClass:
    def main_method(self):
        df = pd.read_csv(r'C:\Users\Ruhi\Workspace\ML_YT\data\HR_comma_sep.csv')
        print(df.head())

        # EDA and visualizations:
        left = df[df.left==1]
        print(left.shape)
        retained = df[df.left==0]
        print(retained.shape)

        print(df.dtypes)
        new_df = df.drop(columns=['Department', 'salary'])
        print(new_df.groupby('left').mean())
        '''
        From above table we can draw following conclusions,

        **Satisfaction Level**: Satisfaction level seems to be relatively low (0.44) in employees leaving the firm vs the retained ones (0.66)
        **Average Monthly Hours**: Average monthly hours are higher in employees leaving the firm (199 vs 207)
        **Promotion Last 5 Years**: Employees who are given promotion are likely to be retained at firm
        '''

        cross_tab = pd.crosstab(df.salary, df.left)

        # Plot the crosstab
        cross_tab.plot(kind='bar')

        # Show the plot

        # Above bar chart shows employees with high salaries are likely to not leave the company
        pd.crosstab(df.Department, df.left).plot(kind='bar')
        # plt.show()
        # From above chart there seem to be some impact of department on employee retention but it is not major hence we will ignore department in our analysis
        '''
        From the data analysis so far we can conclude that we will use following variables as independant variables in our model
        **Satisfaction Level**
        **Average Monthly Hours**
        **Promotion Last 5 Years**
        **Salary**
        '''

        subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]

        # Convert categorical variable to numerical variable:
        salary_dummies = pd.get_dummies(subdf.salary, prefix='salary')
        df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
        print(df_with_dummies.head())
        df_with_dummies.drop(columns='salary', inplace=True)
        print(df_with_dummies.columns)
        X = df_with_dummies
        y = df.left
        print(X.shape)
        print(y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        print(model.predict(X_test))
        print(model.score(X_test, y_test))


if __name__=='__main__':
    MyClass().main_method()