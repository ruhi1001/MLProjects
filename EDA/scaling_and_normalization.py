# ---------------------------------
# Created By : ruhi.ahuja
# Created Date : 8/11/2024
# ---------------------------------

"""
Scaling and Normalization basic difference
"""
import matplotlib.pyplot as plt
import numpy
import numpy as np

# plotting modules
import seaborn as sns

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling
from scipy import stats

# for Box-Cox Transformation

# set seed for reproducibility
np.random.seed(0)


def main_method():
    original_data = numpy.random.exponential(size=1000)
    print(original_data)

    # min-max scaling:
    scaled_data = minmax_scaling(original_data, columns=[0])

    # plot both together to compare:
    _, ax = plt.subplots(1, 2, figsize=(15, 3))
    sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
    ax[0].set_title("Original Data")
    sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
    ax[1].set_title("Scaled Data")
    plt.show()

    # normalize the exponential data with boxcox
    normalized_data = stats.boxcox(original_data)
    print(normalized_data.__sizeof__())

    # plot both together to compare
    _, ax = plt.subplots(1, 2, figsize=(15, 3))
    sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
    ax[0].set_title("Original Data")
    sns.histplot(normalized_data[0], ax=ax[1], kde=True, legend=False)
    ax[1].set_title("Normalized Data")
    plt.show()


if __name__ == "__main__":
    main_method()