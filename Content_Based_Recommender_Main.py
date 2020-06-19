import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#reading post data and user data seperately 
data_frame = pd.read_csv("posts.csv")
user_viewing_data = pd.read_csv("views.csv")

#selecting the features we'll need to use to draw the similarity matrix
features = ['title','category','post_type']

#defining a function for combining the features into one string to get the item profile for post
def Combined_feature_string(row):
	return row['title']+" "+row['category']+" "+row['post_type']

#making sure no empty data points are left in the dataset
for feature in features:
    data_frame[feature] = data_frame[feature].fillna('')

#using our defined function for each row of the data set
data_frame["Combined_feature_string"] = data_frame.apply(Combined_feature_string,axis=1)

#creating new CountVectorizer() object
cv = CountVectorizer() 

#Using fit_transform method from countVectorizer class, to get the count matrix from our feature string
count_matrix = cv.fit_transform(data_frame["Combined_feature_string"])

#cosine similarity matrix from the count matrix we just obtained
cos_sim = cosine_similarity(count_matrix)

#Writing a program to traverse the csv in a record format so that alphanumeric IDs from the dataset can be fetched without indexing error
def csv_to_rec(file_name):
    return np.recfromtxt(file_name, dtype=None, delimiter=',', names=True, encoding='utf-8')

#converting the users csv file to a record and making a dictionary for posts csv
records = csv_to_rec('views.csv')
All_postIDs = data_frame._id
All_Titles = data_frame.title

Content = dict(zip(All_postIDs, All_Titles))

#Defining functions to fetch teh required values

def get_id_from_userid(u_id):
	required_post_id = records.post_id[u_id]
	return required_post_id

def get_title_of_post(id_of_post):
	title_of_post_viewed = Content[id_of_post]
	return title_of_post_viewed

#testing parameter

Test_user_index = 5
Id_of_post = get_id_from_userid(Test_user_index)

postTitle = get_title_of_post(Id_of_post)

similar_posts = list(enumerate(cos_sim[Test_user_index]))
recommended_posts = sorted(similar_posts,key=lambda x:x[1],reverse=True)[1:]

#getting titles of the calculated recommendations
#and Displaying the top five recommended posts based on test user's prefrences 
print('Based on '+postTitle+' you might also like:')
i = 0
for element in range(len(recommended_posts)):
	key = get_id_from_userid(element)
	print('\n')
	print(get_title_of_post(key))
	i = i+1
	if i>5:
		break




