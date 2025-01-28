import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Importing seaborn for making statistical graphics
import numpy as np

sns.set_palette(
    "Paired", 3
)  # Setting the color palette for seaborn plots to "Paired" with 3 different colors
import statsmodels.api as sm  # Importing statsmodels for estimating and testing statistical models
from scipy.stats import (
    linregress,
)  # Importing linregress from scipy.stats for performing linear regression
from scipy.stats import (
    zscore,
)  # Importing zscore from scipy.stats to calculate the z-score for standardization
from scipy.stats import (
    pearsonr,
)  # Importing pearsonr from scipy.stats to calculate the Pearson correlation coefficient
import os  # Importing os for interacting with the operating system


