import streamlit as st
import pickle
import pandas as pd
import io
from sklearn.neighbors import NearestNeighbors
import os
import requests as rq
from streamlit_lottie import st_lottie

#----create browser tab----#
st.set_page_config(page_title="Movie Recommender System", page_icon=":clapper:",  layout="wide"	)


def loadLottieURL(url):
	r = rq.get(url)
	if r.status_code != 200:
		return None
	return r.json()

#-----LOAD ASSETS-----#
lottie_film1 = loadLottieURL("https://assets10.lottiefiles.com/packages/lf20_iv4dsx3q.json")
lottie_film2 = loadLottieURL("https://assets10.lottiefiles.com/private_files/lf30_wqypnpu5.json")
lottie_film3 = loadLottieURL("https://assets10.lottiefiles.com/packages/lf20_ngzwzxib.json")




#----LOGIC FOR COSINE SIMILARITY----
def recommendCosineSimilarity(movie, n):
	movie_idx = movies.index[movies['Movie Title'] == movie][0]
	movie_cosinesimilarity_vector =  cosine_similarity_matrix[movie_idx]
	movie_list = sorted(list(enumerate(movie_cosinesimilarity_vector)), reverse=True, key=lambda x: x[1])

	# movielist = list of tuples -> (movie_index in final_df, similarity)
	print(movie_list[0])
	movie_list = movie_list[0: n]

	# create recommended list of (movie_name, similarity score)
	recommended_list = []

	for i in movie_list:
	    recommended_list.append((movies.iloc[i[0]][0], i))

	return recommended_list




#----LOGIC FOR K NEAREST NEIGHBORS----
def recommendKNN(movie, n):
	movie_idx = movies.index[movies['Movie Title'] == movie][0]

	model_knn = NearestNeighbors(metric="euclidean", algorithm='brute')
	model_knn.fit(cosine_similarity_matrix)

	distances, indices = model_knn.kneighbors(cosine_similarity_matrix[movie_idx].reshape(1, -1), n_neighbors = n)

	recommended_list = []
	for i in range(0, len(distances.flatten())):
	    recommended_list.append((movies.iloc[indices.flatten()[i]][0], distances.flatten()[i]))

	return recommended_list





#----prepare master movie list for the code----#
movies_list = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies_list)

#----take cos similarity matrix into account----#
cosine_similarity_matrix = pickle.load(open('cosine_similarity_matrix.pkl', 'rb'))

movies_list = movies_list['Movie Title'].values



#-------DRIVER CODE-------#
def main():


	#-----PREPARE HEADERS-----#
	engine_option = ""

	#----n = number of movies----#
	n = 0

	with st.container():
		left, right = st.columns(2)

		with left:
			# st.subheader("Ready, Set, Recommend! :sunglasses:")
			st.title('Enter your Tang:fire: :fire:')
			#----create radio button----#
			engine_option = st.radio('Select your Chef Recommender Engine: ', ('Cosine Similarity', 'K Nearest Neighbors'))

			#----create slider----#
			# range = (1, 50)
			n = st.slider('Enter number of movies to eat: ', 1, 50, 25)

		with right:
			st_lottie(lottie_film3, height=500, width=550)



	st.subheader("Ready, Set, Recommend! :sunglasses:")
	option = st.selectbox(
	'How we may entertain you?',
	movies_list)

	btn = st.button('Let\'s Go')


	#---initialize recommendation movie list with empty list---#
	recommendations = []

	with st.container():
		left_column, mid, right_column = st.columns(3)

		if btn:
			if engine_option == 'Cosine Similarity':
				recommendations = recommendCosineSimilarity(option, n)
			elif engine_option == 'K Nearest Neighbors':
				recommendations = recommendKNN(option, n)


			#----display recommended movies----#
			with left_column:
				for i in range(0, len(recommendations)//2):
					st.write(recommendations[i][0])
			
			with mid:
				for i in range(len(recommendations)//2, len(recommendations)):
					st.write(recommendations[i][0])

		#----display bottom animation----#
		with right_column:
			st_lottie(lottie_film1, height=400, width=400)
			st_lottie(lottie_film2, height=400, width=400)



# CODE STARTS HERE
if __name__ == "__main__":
	main()
	