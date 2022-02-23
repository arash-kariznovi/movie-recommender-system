#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

r_cols = ["user_id","movie_id","rating"]
ratings = pd.read_csv("datasets/u.data",sep="\t",names=r_cols,usecols=range(3))

m_cols = ["movie_id", "title"]
movies = pd.read_csv("datasets/u.item",sep="|",names=m_cols,usecols=range(2),encoding="ISO-8859-1")

# merge ratings and movies tables
df = pd.merge(ratings,movies)

df.head()


# In[4]:


# make a table 
# rows : users and cols: movies and values: ratings 
pivot = df.pivot_table(index="user_id",columns="title", values="rating")
pivot.head()


# In[8]:


# we want to recommand based on all cols
corrMatrix = pivot.corr()
corrMatrix


# In[11]:


# just based on pairs that were rated over 100 times
corrMatrix_2 = pivot.corr(method="pearson",min_periods=100)
corrMatrix_2.head()


# In[17]:


# a random user that we want to recommand movies to him/her
myRatings = pivot.loc[0].dropna()


# In[55]:


similarCandidates = pd.Series()

for i in range(len(myRatings)):
    
    print("Finding Similars to ", myRatings.index[i],"...")
#     retrieve similars from corrMatrix
    sims = corrMatrix_2[myRatings.index[i]].dropna()
    sims = sims.map(lambda x : x*myRatings[i])
    similarCandidates = similarCandidates.append(sims)
    

similarCandidates.sort_values(inplace=True,ascending=False)
similarCandidates.head(10)
# we have similar movies in the films the man has seen..
# so we will see some movies more than once for each of them..


# In[59]:


# to remove previous duplication add correlations
similarCandidates = similarCandidates.groupby(similarCandidates.index).sum()


# In[60]:


similarCandidates.sort_values(inplace=True, ascending=False)
similarCandidates.head(10)


# In[63]:


# filter out movies that user has seen before
recommandedMovies = similarCandidates.drop(myRatings.index)
recommandedMovies.head(5)


# In[ ]:




