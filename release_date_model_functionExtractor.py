#!/usr/bin/env python
# coding: utf-8

# # **Movie Recommender on the basis of Release Date of Movie(s)**

# In[1]:


import numpy as np
import pandas as pd


# # **1. create 2 dataframes netflixMovie_df and imdbMovie_df**

# **preprocessing netflix dataframe**

# In[2]:


netflixMovie_df = pd.read_csv('Netflix_Dataset_Movie.csv')
netflixRating_df = pd.read_csv('Netflix_Dataset_Rating.csv')


# In[3]:


# we don't need rating dataframe as there is no year of launch of movies
netflixRating_df.head(1)


# In[4]:


netflixMovie_df


# In[5]:


# we need movie dataFrame of only netflixMovie_df = ['movie_names', 'release date']

netflixMovie_df = netflixMovie_df[['Name', 'Year']]
netflixMovie_df


# In[6]:


# convert 'Name' -> 'Movie_Title'
# convert 'Year' -> 'Released_Year'

netflixMovie_df = netflixMovie_df.rename(columns={'Name': 'Movie_Title'})
netflixMovie_df = netflixMovie_df.rename(columns={'Year': 'Released_Year'})


# In[7]:


netflixMovie_df.head(1)


# **preprocessing imdb dataframe**

# In[8]:


imdbMovie_df = pd.read_csv('imdb_top_1000.csv')


# In[9]:


imdbMovie_df.head(1)


# In[10]:


# we need movie dataFrame of only imdbMovie_df = ['movie_names', 'release date']

imdbMovie_df = imdbMovie_df[['Series_Title', 'Released_Year']]


# In[11]:


imdbMovie_df.head(1)


# In[12]:


# convert 'Series_Title' -> 'Movie_Title'

imdbMovie_df = imdbMovie_df.rename(columns={'Series_Title': 'Movie_Title'})


# In[13]:


imdbMovie_df.head(1)


# In[14]:


# check if there is some incorrect string value in imdbMovie_df & store it in movie_idx

def checkIncorrectValues():
    movie_idx = -1
    for i in range(len(imdbMovie_df['Released_Year'])):
        if(imdbMovie_df['Released_Year'].iloc[i] != 'PG'):
            imdbMovie_df['Released_Year'].iloc[i] = int(imdbMovie_df['Released_Year'].iloc[i])
        else:
            movie_idx = i
    return movie_idx


# In[15]:


obtained_movie_idx = checkIncorrectValues()


# In[16]:


# check the incorrect 'Movie_Title' value in imdbMovie_df

imdbMovie_df['Movie_Title'][obtained_movie_idx]


# In[17]:


# fill the movie's 'Released_Year' w/ movie's release date

imdbMovie_df['Released_Year'][obtained_movie_idx] = 1970


# # **2. Review of netflixMovie_df and imdbMovie_df**

# In[18]:


netflixMovie_df


# In[19]:


imdbMovie_df


# In[20]:


print('shape of netflixMovie_df  : ', netflixMovie_df.shape)
print('shape of imdbMovie_df     : ', imdbMovie_df.shape)


# # **3. merge netflixMovie_df & imdbMovie_df to form yearMovie_df** 

# In[21]:


# now merge the two dataframes into yearMovie_df

yearMovie_df = pd.concat([netflixMovie_df, imdbMovie_df], axis=0)


# In[22]:


yearMovie_df


# In[23]:


# sort yearMovie_df on the basis of 'Released_Year'

yearMovie_df = yearMovie_df.sort_values(by=["Released_Year"])


# In[24]:


yearMovie_df


# In[25]:


# add 'User_id' to yearMovie_df

yearMovie_df['User_Id'] = [i for i in range(len(yearMovie_df.index))]


# In[26]:


yearMovie_df


# In[27]:


arr = yearMovie_df.to_numpy()


# In[28]:


arr


# # **4. Create a pivot table movieUser_df**

# In[29]:


yearMovie_df.drop_duplicates(subset='Movie_Title', keep = 'first', inplace = True)


# In[30]:


yearMovie_df


# In[31]:


# drop first 12867 rows
N = 12867
yearMovie_df = yearMovie_df.iloc[N: , :]


# In[32]:


yearMovie_df


# In[33]:


yearMovie_df['Movie_Title'][0]


# In[34]:


# reserialize 'User_Id'

yearMovie_df['User_Id'] = [i for i in range(len(yearMovie_df.index))]


# In[35]:


movieUser_df = pd.pivot_table(yearMovie_df, index='Movie_Title', columns='User_Id', values='Released_Year')
# yearMovie_df


# In[36]:


movieUser_df


# In[37]:


# drop last 
N = 3000
movieUser_df = movieUser_df.iloc[: , :-N]


# In[38]:


movieUser_df


# # **5. make movieUser_df sparse**

# In[39]:


m = np.random.randint(low=-300, high=6, size=(5000, 2000), dtype=int)


# In[40]:


movieUser_df.index[1]


# In[41]:


# create a list containing all movie names

movieList=[]
for i in range(len(movieUser_df.index)):
    movieList.append(movieUser_df.index[i])


# In[42]:


# store the values in userMovie_df

movieUser_df = pd.DataFrame(m, index=movieList)


# In[43]:


# make it sparse
movieUser_df[movieUser_df < 1] = 0


# In[44]:


# fill all 'nan' values with 0

movieUser_df.fillna(0)


# # *6. Core Logic of Recommender System using Binary Search*

# In[45]:


# this function returns the last index of highest valued rating, and the correponding movie name 

def lastIndexOfTopRatedMoviesByUserX(user_series_of_movies, rating, l, h):
    ans_idx = -1
    while l <= h:
        mid = l + (h-l)//2
        if user_series_of_movies[mid] >= rating:
            ans_idx = mid
            last_coordinated_movie = user_series_of_movies.index[mid]
            l = mid + 1
        else:
            h = mid - 1 
            
    return ans_idx, last_coordinated_movie


# In[103]:


# n = number of movies per top ratings of user
# u = 'User_id'
# rating = lowest best rating -> [1, 5]
# last_coordinated_movie -> name of last highly rated movie by the user-X
n = 5
u = 0
rating = 4
# last_coordinated_movie
movieUser_df = movieUser_df.sort_values(by=[0], axis=0, ascending=False)


# In[104]:


movieUser_df


# In[105]:


if 'Avatar' in movieUser_df.index:
    print(True)


# In[106]:


movieUser_df[u].fillna(value=0, inplace=True)


# In[107]:


# call the lastIndexOfTopRatedMoviesByUserX
l = 0
h = len(movieUser_df.columns)-1
idx, last_coordinated_movie = lastIndexOfTopRatedMoviesByUserX(movieUser_df[u], rating, l, h)


# In[108]:


print("the index of last highly rated movie by the user-X: ", idx)
print("the name of last highly rated movie by the user-X: ", last_coordinated_movie)


# In[109]:


# verify the name of the movie received

movieUser_df[0].index[idx]


# In[110]:


movieUser_df[0][idx]


# In[112]:


movieUser_df[0][idx+1]


# In[113]:


# Hence the calculation is correct


# # ***7. Recommendation code returning a list of movies***

# In[114]:


# Helper function to return whether the movie is rated or not to avoid recommending already rated movie

def isRated(movie_name):
    if movieUser_df[u][movie_name] > 0:
        return True
    return False


# In[115]:


# This function returns:-
# 1. the recommended movie list
# 2. the year of the movie in year sorted yearMovie_df OR, the year of last_coordinated_movie for verification

def recommendMovies(u, n, idx, last_coordinated_movie, yearMovie_df_array):
    movie_list = []
    pivot_movie_idx = -1
    pivot_movie_year = -1
    for i in range(len(yearMovie_df_array)):
        j = len(yearMovie_df_array) - i - 1
        if(yearMovie_df_array[i][0] == last_coordinated_movie):
            pivot_movie_idx = i
            pivot_movie_year = yearMovie_df_array[i][1]
            print(yearMovie_df_array[i][0])
            break
        if(yearMovie_df_array[j][0] == last_coordinated_movie):
            pivot_movie_idx = j
            pivot_movie_year = yearMovie_df_array[j][1]
            print(yearMovie_df_array[j][0])
            break
            
    
    # store closest movies greater than or equal to current year 
    right_movie_cnt = 0
    right_starter_idx = pivot_movie_idx + 1
    while right_movie_cnt < n:
        if(right_starter_idx > len(yearMovie_df_array)-1):
            break
            
        if(isRated(yearMovie_df_array[right_starter_idx][0]) == False):
            
            movie_list.append( (yearMovie_df_array[right_starter_idx][0], yearMovie_df_array[right_starter_idx][1]) )
            right_movie_cnt += 1
            
        right_starter_idx += 1
            
            
    
    # store closest movies less than or equal to current year 
    left_movie_cnt = 0
    left_starter_idx = pivot_movie_idx - 1
    while left_movie_cnt < n:
        if(left_starter_idx == 0):
            break
            
        if(isRated(yearMovie_df_array[left_starter_idx][0]) == False):
            
            movie_list.append( (yearMovie_df_array[left_starter_idx][0], yearMovie_df_array[left_starter_idx][1]) )
            left_movie_cnt += 1
        
        left_starter_idx -= 1
        
            
    return movie_list, pivot_movie_year


# In[116]:


# convert yearMovie_df -> numpy array

yearMovie_df_array = yearMovie_df.to_numpy()


# In[117]:


len(yearMovie_df_array)


# In[118]:


yearMovie_df_array


# In[119]:


# movieUser_df[0]['Hamilton']


# In[120]:


# idx variable contains the last index of highest valued rating
# last_coordinated_movie contains the name of the last highly rated movie by user-X
# pivot_movie_year = the year of movie obtained as last_coordinated_movie

recommendations, pivot_movie_year = recommendMovies(u, n, idx, last_coordinated_movie, yearMovie_df_array)


# In[125]:


recommendations[:3]


# In[122]:


# validate the year of obtained movie with obve recommended movies
pivot_movie_year

# the movie year which was predicted for the selected user it is recommending the movies closest to this year 
# and here it is verified


# In[123]:


# Here is the important part
# we can verify that all the recommendations above are "NON-RATED"
# if the rating value in movieUser_df = 0 of any user 'u' (for any movie above) 
# then that must mean the movie is of course NON-RATED

movieUser_df[u]['Jay-Z: Fade to Black']


# In[124]:


# verify the recommendations are unrater

movieUser_df[u]['Jay-Z: Fade to Black']


# In[ ]:




