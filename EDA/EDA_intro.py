# ---------------------------------
# Created By : ruhi.ahuja
# Created Date : 26/10/2024
# ---------------------------------

"""
*** File details here
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class EDA:
    def main_method(self):
        # Fetching data:
        df = pd.read_csv('EDA_into_data.csv')
        print(df.head())
        print(df.tail())

        # Checking datatypes:
        print(df.dtypes)

        # Dropping irrelevant columns :
        df = df.drop(['Engine Fuel Type', 'Market Category', 'Vehicle Style', 'Popularity', 'Number of Doors',
                      'Vehicle Size'], axis=1)
        print(df.head())

        # Renaming the columns:
        df = df.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders",
                                "Transmission Type": "Transmission", "Driven_Wheels": "Drive Mode",
                                "highway MPG": "MPG-H", "city mpg": "MPG-C", "MSRP": "Price"})
        print(df.head(5))

        # Dropping Duplicate rows:
        print(df.shape)
        duplicate_row_df = df[df.duplicated()]  ## Learnt
        print('Number of duplicate rows: ', duplicate_row_df.shape)
        df = df.drop_duplicates()  ## Learnt
        print(df.shape)
        print(df.count())  ## Learnt

        # Dropping the missing values:
        print(df.isnull().sum())
        df = df.dropna()
        print(df.count())
        print(df.isnull().sum())

        # Detecting Outliers:
        '''
        The outlier detection and removing that I am going to perform is called IQR score technique. 
        Often outliers can be seen with visualizations using a box plot. 
        Shown below are the box plot of MSRP, Cylinders, Horsepower and EngineSize.
        '''
        sns.boxplot(x=df['Price'])
        sns.boxplot(x=df['HP'])
        sns.boxplot(x=df['Cylinders'])
        # plt.show()

        df_numeric = df.select_dtypes(include=['number'])
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        print(IQR)
        df_numeric = df_numeric[~((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR)))]
        print(df_numeric.shape)

        # Plotting:
        # Histogram:
        '''
        Histogram refers to the frequency of occurrence of variables in an interval. 
        In this case, there are mainly 10 different types of car manufacturing companies, 
        but it is often important to know who has the most number of cars. 
        To do this histogram is one of the trivial solutions which lets us know the total number of car manufactured by a different company.
        '''
        df['Make'].value_counts().nlargest(40).plot(kind='bar', figsize=(10, 5))
        plt.title('Number of cars by make')
        plt.ylabel('Number of cars')
        plt.xlabel('MAke')
        # plt.show()

        # Heatmaps:
        '''
        Heat map is a type of plot which is necessary when we need to find the dependent variables.
        One of the best way to find the relationship between the features can be done using heat maps. 
        In the below heat map we know that the price feature depends mainly on the Engine Size, Horsepower, and Cylinders.
        '''
        plt.figure(figsize=(10, 5))
        c = df_numeric.corr()
        sns.heatmap(c, cmap="BrBG", annot=True)
        print(c)
        # plt.show()

        # Scatterplot:
        '''
        We generally use scatter plots to find the correlation between two variables. 
        Here the scatter plots are plotted between Horsepower and Price and we can see the plot below. 
        With the plot given below, we can easily draw a trend line. 
        These features provide a good scattering of points.
        '''
        _, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['HP'], df['Price'])
        ax.set_xlabel('HP')
        ax.set_ylabel('Price')
        plt.show()


if __name__=='__main__':
    EDA().main_method()