#!/usr/bin/env python
# coding: utf-8

# # Movie Recommender System using Unsupervised Learning and NLP

# In[1]:


# ignore unnecessary warnings by libraries
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd


# # 

# # DataFrame Format :-
# {'Movie_Title': [list_of_movies], 'Genres': [list_of_genres], 'Director': 'director_name', 'Cast': [top_3_cast], 'tagline': 'tagline_description'}

# # Preprocessing tmdb dataframe

# In[3]:


tmdbMovies_df = pd.read_csv('current_datasets/tmdb_5000_movies.csv')
tmdbCredits_df = pd.read_csv('current_datasets/tmdb_5000_credits.csv')


# In[4]:


tmdbMovies_df.head(1)


# In[5]:


tmdbCredits_df.head(1)


# In[6]:


# the director in tmdbCredits_crew is given at crew column
# tmdbCredits_df['crew'][0]


# In[7]:


final_df1 = tmdbMovies_df.merge(tmdbCredits_df, on='title')


# In[8]:


# we need title, genres, director, cast, tagline

final_df1 = final_df1[['title', 'genres', 'overview', 'tagline', 'keywords', 'cast', 'crew']]


# In[9]:


#now check if there is any sort of missing data

final_df1.isnull().sum()


# In[10]:


final_df1=final_df1[final_df1['overview'].notna()]


# In[11]:


final_df1.isnull().sum()


# In[12]:


# fill all nan values in 'tagline' with empty string ""

final_df1 = final_df1.replace(np.nan, '', regex=True)


# In[13]:


final_df1.isnull().sum()


# In[14]:


final_df1.head(1)


# # 

# # Preprocessing the-movies-dataframe

# In[15]:


movies_df = pd.read_csv('current_datasets/movies_metadata.csv', low_memory=False)
credits_df = pd.read_csv('current_datasets/credits.csv', low_memory=False)


# In[16]:


credits_df.head(1)


# In[17]:


movies_df.head(1)


# In[18]:


final_df2 = pd.concat([movies_df, credits_df], axis=1)


# In[19]:


final_df2


# In[20]:



final_df2 = final_df2[['title', 'genres', 'overview', 'tagline', 'belongs_to_collection', 'cast', 'crew']]
final_df2 = final_df2.rename(columns={'belongs_to_collection': 'keywords'})


# In[21]:


final_df2=final_df2[final_df2['title'].notna()]


# In[22]:


final_df2


# In[23]:


final_df2.shape


# In[24]:


# double check and drop all nan values
final_df2 = final_df2.dropna()
final_df2


# # 

# # Cancatenate:   final_df1   +   final_df2  =  final_df

# In[25]:


final_df1.shape


# In[26]:


final_df2.shape


# In[27]:


final_df = pd.concat([final_df2, final_df1], axis=0, ignore_index=True)


# In[28]:


final_df.shape


# In[29]:


final_df


# In[30]:


final_df = final_df.dropna()


# In[31]:


final_df.shape


# In[32]:


# drop first 12867 rows
# N = 39312
# final_df = final_df.iloc[:-N]
# final_df


# # 

# # Editing genres

# In[33]:


# hence there are no 'nan' values in final_df
final_df.isnull().sum()


# In[34]:


# remove all empty genre columns
final_df.iloc[0].genres


# In[35]:


# it is string so convert it into integer using --> 'ast' module

import ast

#helper function to return list of genres, keywords etc.
def convert(obj):
    List = []
    for i in ast.literal_eval(obj):
        List.append(i['name'])
    return List


# In[36]:


final_df['genres'] = final_df['genres'].apply(convert)


# In[37]:


final_df.head(3)


# # 

# # Editing cast

# In[38]:


#helper function to return first 3 cast names of each movie

def convertCast(obj):
    List = []
    counter = 0;
    for i in ast.literal_eval(obj):
        if counter < 3:
            List.append(i['name'])
            counter += 1
        else:
            break
    return List


# In[39]:


#store result in movies['cast']
final_df['cast'] = final_df['cast'].apply(convertCast)


# In[40]:


final_df.head(3)


# # 

# # Editing crew

# In[41]:


#we only need director for our purpose

#helper function to fetch director
def fetch_director(obj):
    List = []
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            List.append(i['name'])
            break
    return List


# In[42]:


final_df['crew'] = final_df['crew'].apply(fetch_director)


# In[43]:


final_df.head(3)


# # 

# # Editing overview

# In[44]:


# check overview data format
final_df['overview'][0]


# In[45]:


# convert this string into list
final_df['overview'] = final_df['overview'].apply(lambda x: x.split())


# In[46]:


final_df.head(3)


# # 

# # Remove spaces between words of same entity

# In[47]:


# final_df['genres'] = final_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
# final_df['cast'] = final_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
# final_df['crew'] = final_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])


# In[48]:


final_df.head(1)


# # 

# # Create a final Data Frame Composed only of 3 columns
# # 'Movie-Title' & 'Movie-elements'

# In[49]:


type(final_df['tagline'][0])


# In[50]:


final_df['tagline'] = final_df['tagline'].apply(lambda x: x.split())


# In[51]:


# final_df.iloc[:6] = final_df['overview'] + final_df['genres']  + final_df['cast'] + final_df['crew']
lst = []
lst = final_df['overview']  + final_df['tagline'] + final_df['genres'] + final_df['cast'] + final_df['crew']


# In[52]:


lst


# In[53]:


final_df['Movie Elements'] = [i for i in lst]
final_df['Movie Elements'] = final_df['Movie Elements'].apply(lambda x: " ".join(x))


# In[54]:


final_df.head(1)


# In[55]:


# Now, we have ['overview', 'tagline', 'genres', 'cast', 'crew'] in one seperate column --> 'Movie Elements'!!

final_df = final_df[['title', 'Movie Elements']]


# In[56]:


# Demo
final_df['Movie Elements'][0]


# In[57]:


final_df.head(5)


# In[58]:


# rename 'title' --> 'Movie Title'
final_df = final_df.rename(columns={'title': 'Movie Title'})


# In[59]:


final_df = final_df.drop_duplicates(keep='first')


# In[60]:


final_df


# In[61]:


# reset index values for better accuracy by model
final_df = final_df.reset_index(drop=True)


# In[62]:


final_df


# # 

# # Step-1: Stemming (text cleaning) using nlp

# In[63]:


import nltk
from nltk.stem.snowball import SnowballStemmer


# In[64]:


stemmer_tool = SnowballStemmer(language="english")


# In[65]:


# Helper function to facilitate stemming

def stemMovieElements(text):
    text = list(text.split(" "))
    stemmed_text = [stemmer_tool.stem(i) for i in text]
    return ' '.join(stemmed_text)


# In[66]:


print("Text before stemming:\n",final_df['Movie Elements'][0])


# In[67]:


text = stemMovieElements(final_df['Movie Elements'][0])
print("Text after stemming:\n", text)


# In[68]:


# now apply stemming to all of the 'Movie Elements'
final_df['Movie Elements'] = final_df['Movie Elements'].apply(stemMovieElements)


# In[69]:


final_df.head(5)


# # 

# ## Introducing Term Frequency-Inverse Document Frequency (Tf-Idf)
# ### (a text vectorization technique)

# ### -- TF (term frequency): records frequency of words in a document. It then normalizes that frequency
# 
# ### normalization formula   : TF(word: w) = (# of times w occurs in a text) / (total # of words in the text)  
# 
# 
# 
# 
# ### -- IDF (inverse document frequency): computes importance of a word in a document. It is the frequency of docs in the corpus containing the word. This frequency is then inversed. It prioritizes meaningful, less frequently occuring words in the doc and scales down the weight of frequently occuring needless words eg. stop words.
# 
# ### formula: IDF(word: w) = log[(total # of texts) / (# of texts containing w in corpus C)]
# 
# 
# 
# ### -- TF-IDF: In this approach weight of a word in a text/document is first calculated by (TF), then (IDF) of the word in the corpus is calculated. product(tf(w), idf(w)) = tf-idf(w)
# 
# ### formula: TF-IDF(word: w, text: t, corpus: C) = TF(w, t) . IDF(w, t, C)
# 
# ### meaningful words in a text -> higher tf-idf.  less meaningful words in text -> almost 0 tf-idf

# In[ ]:





# # Step-2: Text Vectorization

# In[70]:


# using TF-IDF technique
from sklearn.feature_extraction.text import TfidfVectorizer


# In[71]:


# initialize vectorizer
tfidfVectorizer = TfidfVectorizer(max_features=6612, stop_words=None)


# In[72]:


# create a matrix to store the vectorized data
tfidf_matrix  = tfidfVectorizer.fit_transform(final_df['Movie Elements'])


# In[73]:


tfidf_matrix = tfidf_matrix.toarray()


# In[75]:


# sparse tfidf matrix. because this matrix stores unique words in the Corpus(C) and counts their frequency in a mathematical
# way. Giving importance to less occuring meaningful words
tfidf_matrix


# In[91]:


# it is sparse but its most heavy word is ccalculated mathematically using log
print(max(tfidf_matrix[20]))


# In[92]:


tfidf_matrix.shape


# # 

# # Step-3: calculate Cosine Similarity between a movie to every other movie

# In[93]:


from sklearn.metrics.pairwise import cosine_similarity


# In[94]:


# lets get our cosine similarity matrix
cosine_similarity_matrix = cosine_similarity(tfidf_matrix)


# In[95]:


cosine_similarity_matrix.shape


# In[96]:


cosine_similarity_matrix[0]


# In[97]:


# create a fn to return list('movie name', '(idx value', similarity score))


# In[117]:


def recommendMovies(movie_name):
    movie_idx = final_df.index[final_df['Movie Title'] == movie_name][0]
    movie_cosinesimilarity_vector =  cosine_similarity_matrix[movie_idx]
    movie_list = sorted(list(enumerate(movie_cosinesimilarity_vector)), reverse=True, key=lambda x: x[1])
    
    # movielist = list of tuples -> (movie_index in final_df, similarity)
    print(movie_list[0])
    movie_list = movie_list[0: 100]
    
    # create recommended list of (movie_name, similarity score)
    recommended_list = []
    
    for i in movie_list:
        recommended_list.append((final_df.iloc[i[0]][0], i))
        
    return recommended_list


# In[99]:


# print(final_df.index[final_df['Movie Title'] == 'Iron Man'])
final_df['Movie Title'][4]


# In[100]:


final_df.iloc[573][1]


# In[101]:


movie_idx = final_df.index[final_df['Movie Title'] == 'Toy Story']
movie_cosinesimilarity_vector =  tfidf_matrix[movie_idx]


# In[102]:


# the cosing similarity vector of a movie is sparse by nature because a lot of movies are unrelated to one movie
movie_cosinesimilarity_vector


# # judgement time!!

# In[118]:


# some sample movies to test on
# 'Avatar', 'Iron Man', 'Predator', 'Batman', 'Toy Story', 'Jurassic Park', 'Spider-Man', 'Thor'
# 'The Avengers', 'Dracula', 'Godzilla', 'Warcraft', 'X-Men', 'The Matrix', 'Dead Man Down'
recommended_list = recommendMovies('Iron Man')
recommended_list


# In[104]:


final_df


# In[119]:


# to find a particular movie in corpus for further inspection
movie_idx = final_df.index[final_df['Movie Title'] == 'Iron Man 3'][0]
print(movie_idx)


# In[ ]:





# In[ ]:




