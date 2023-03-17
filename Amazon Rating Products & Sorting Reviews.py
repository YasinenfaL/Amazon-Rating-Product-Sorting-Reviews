###################################################
# Rating Product & Sorting Reviews in Amazon
###################################################

# Libraries
import pandas as pd
import numpy as np
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def check_df(dataframe, head=5):
    print("############### Shape ################")
    print(dataframe.shape)
    print("########### Types ###############")
    print(dataframe.dtypes)
    print("########### Head ###############")
    print(dataframe.head(head))
    print("########### Tail ###############")
    print(dataframe.tail(head))
    print("########### NA ###############")
    print(dataframe.isnull().sum())
    print("########### Quantiles ###############")
    print(dataframe.describe([0, 0.25, 0.50, 0.75]).T)


###################################################
# TASK 1: Calculate Average Rating Based on Current Comments and Compare with Existing Average Rating.
###################################################

###################################################
# Step 1: Read the Data Set and Calculate the Average Score of the Product.
###################################################
df_ = pd.read_csv("datasets/Ölçümleme/amazon_review.csv")
df = df_.copy()
check_df(df)

df["overall"].mean()


###################################################
# Step 2: Calculate the Weighted Average of Score by Date.
###################################################

def time_based_averege(dataframe, w1=30, w2=28, w3=22, w4=20):
    return dataframe.loc[(dataframe["day_diff"] <= 281, "overall")].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 281 & (dataframe["day_diff"] <= 431), "overall")].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 431 & (dataframe["day_diff"] <= 601), "overall")].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 601, "overall")].mean() * w4 / 100


def time_based_weighted_average(dataframe, w1=30, w2=28, w3=22, w4=20):
    return dataframe.loc[dataframe["day_diff"] <= 281, "overall"].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 281) & (dataframe["day_diff"] <= 431), "overall"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 431 & (dataframe["day_diff"] <= 601), "overall")].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 601), "overall"].mean() * w4 / 100


time_based_weighted_average(df)

time_based_averege(df)

###################################################
# Step 3: The first average score should be compared with the weighted score according to the date to be obtained.
###################################################
df["overall"].mean()
print(df.loc[df["days"] <= 283, "overall"].mean())
print(df.loc[(df["days"] > 283 & (df["days"] <= 433)), "overall"].mean())
print(df.loc[(df["days"] > 433 & (df["days"] <= 603)), "overall"].mean())
print(df.loc[(df["days"] > 603, "overall")].mean())

###################################################
# Task 2: Specify 20 Reviews for the Product to be Displayed on the Product Detail Page.
###################################################


###################################################
# Step 1. Generate the helpful_no variable
###################################################
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


###################################################
# Step 2. Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores and Add to Data
###################################################

def score_pos_neg_diff(helpful_yes, helpful_no):
    return helpful_yes - helpful_no


def score_average_rating(helpful_yes, helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    return helpful_yes / (helpful_yes + helpful_no)


def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score

    - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
    - The score to be calculated is used for product ranking.
    - Note:
    If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and can be made to conform to Bernoulli.
    This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

    parameters
    ----------
    helpful_yes: int
        up count
    helpful_no: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


check_df(df)

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_ye  s"],
                                                                 x["helpful_no"]), axis=1)

##################################################
# Step 3. Identify 20 Interpretations and Interpret Results.
###################################################
df.sort_values("score_pos_neg_diff", ascending=False).head(20)

df.sort_values("score_average_rating", ascending=False).head(20)

df.sort_values("wilson_lower_bound", ascending=False).head(20)
