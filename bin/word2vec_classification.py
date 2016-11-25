
# coding: utf-8

# # word2vec and classification

# GitHub repository with this analysis: https://github.com/JungeAlexander/kaggle-amazon-fine-food-reviews
# 
# Some inspiration can be found here:
# 
# - https://www.kaggle.com/c/word2vec-nlp-tutorial
# - https://www.kaggle.com/gpayen/d/snap/amazon-fine-food-reviews/building-a-prediction-model/notebook
# - https://www.kaggle.com/inspector/d/snap/amazon-fine-food-reviews/word2vec-logistic-regression-0-88-auc/notebook

# In[1]:

import numpy as np
import pandas as pd
#from sklearn import
import sqlite3


# # Labelling good and bad reviews
# 
# To obtain binary class labels, reviews with 4-5 stars are considered good and assigned a '1' class label while reviews with less than 3 stars are considered bad and assigned a '0' class label. Reviews with 3 stars are ignored.

# In[2]:

connection = sqlite3.connect('../input/database.sqlite')
reviews = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3""", connection)


# In[3]:

reviews['Class'] = 1 * (reviews['Score'] > 3)


# In[4]:

reviews.head(n=2)


# ## Split into training and test sets
# 
# Split the data set into training and test sets. To ensure indendence of training and test set and generalizability of the model, make sure that no product (identified by `ProductId`) and no user (identified by `UserId`) is present in both training and test set.
# 
# This is implemented by first sorting the data set by `ProductId`, splitting into equally sized training and test set (no shuffling!) and lastly removing any review from the test set where either user or product ID also appears in the training set. Of course, this will make the training and test sets unequally sized but ¯\\\_\_(ツ)\_\_/¯

# In[5]:

reviews.sort_values('ProductId', axis=0, inplace=True)


# In[6]:

train_size = int(len(reviews) * 0.5)
train_reviews = reviews.iloc[:train_size,:]
test_reviews = reviews.iloc[train_size:,:]


# In[7]:

test_remove = np.logical_or(test_reviews['ProductId'].isin(train_reviews['ProductId']),
                          test_reviews['UserId'].isin(train_reviews['UserId']))
test_reviews = test_reviews[np.logical_not(test_remove)]


# In[8]:

print('Training set contains {:d} reviews.'.format(len(train_reviews)))
print('Test set contains {:d} reviews ({:d} removed)'.format(len(test_reviews), sum(test_remove)))


# # TODO
# 
# - check class proportions in train and test set
# - train word embedding
# - classify
