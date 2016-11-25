
# coding: utf-8

# # word2vec and classification

# GitHub repository with this analysis: https://github.com/JungeAlexander/kaggle-amazon-fine-food-reviews
# 
# Some inspiration can be found here:
# 
# - https://www.kaggle.com/c/word2vec-nlp-tutorial
# - https://www.kaggle.com/gpayen/d/snap/amazon-fine-food-reviews/building-a-prediction-model/notebook
# - https://www.kaggle.com/inspector/d/snap/amazon-fine-food-reviews/word2vec-logistic-regression-0-88-auc/notebook

# In[2]:

import pandas as pd
import sqlite3


# # Labelling good and bad reviews
# 
# To obtain binary class labels, reviews with 4-5 stars are considered good and assigned a '1' class label while reviews with less than 3 stars are considered bad and assigned a '0' class label. Reviews with 3 stars are ignored.

# In[3]:

connection = sqlite3.connect('../input/database.sqlite')
reviews = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3""", connection)


# In[9]:

reviews['Class'] = 1 * (reviews['Score'] > 3)


# In[10]:

reviews.head(n=2)


# ## Split into train and test sets
# 
# Split the data set into about equally sized train and test sets. To ensure indendence of train and test set and generalizability of the model, make sure that no product (identified by `ProductId`) and no user (identified by `UserId`) is present in both train and test set.

# In[ ]:




# - train word embedding
# - classify
