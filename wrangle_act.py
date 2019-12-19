#!/usr/bin/env python
# coding: utf-8

# ## Wrangle WeRateDogs Twitter data

# #### In this project I will wrangling data which consists of gathering, assessing  and cleaning, then analyses and visualizations Twitter data using three data set , Enhanced Twitter Archive ,Image Predictions File and Tweet JSON Data using API.

# In[1]:


#Import the libraries that we will need in this project
import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import json
import requests
import tweepy


# ## Gathering Data

# ### twitter archive

# In[2]:



# Read twitter archive in csv file 
twitter_archive = pd.read_csv('twitter-archive-enhanced.csv')


# ### image predictions

# In[3]:


# Use requests library to download image prediction tsv file from a website
image_prediction_url=" https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
response = requests.get(image_prediction_url)

with open('image_predictions.tsv', mode='wb') as file:
    file.write(response.content)


# In[4]:


# Read in tsv file as a Pandas DataFrame    
image_predictions = pd.read_csv('image_predictions.tsv', sep='\t')


# ### json data
# 
# <li> Query the Twitter API using Tweepy library to get JSON data for each of the tweets in the WeRateDogs Twitter archive.

# In[5]:


#Personal API keys, secrets, and tokens
consumer_key = 'Z7EQF9sFZ6USZuoFVqSe36Fqj'
consumer_secret = '6l3GjgpE1MEL43o3BNl5qUxwZzIRIaZPqIOfcErYpXfDzaWyz5'
access_token = '2695424372-T5Y56hW9sIbjD7rZ2NLHWT2xhUiEDRSdg613BJD'
access_secret = 'iVA14fQLlH90AWXJGoBmMD19lkXLURtQYooWufAAynKEd'


# In[6]:


#creating an OAuthHandler then access to the Twitter API then create an API object that we are going to use it to fetch the tweets
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth , wait_on_rate_limit=True)


# In[7]:


list_t=[]
with open('tweet_json.txt', 'a', encoding='utf8') as f:
    for tweet_id in twitter_archive['tweet_id']:
        try:
            tweet = api.get_status(tweet_id, tweet_mode='extended')
            json.dump(tweet._json, f)
          f.write('\n')
        except:
            list_t.append(tweet_id)
          


# In[8]:


# open JSON as file and append required fields to array

tweets_list =[]

with open('tweet_json.txt') as json_file:
    for line in json_file:
    
        tweets_dict = {}
        tweets_json = json.loads(line)
        
        try:
            tweets_dict['tweet_id'] = tweets_json['extended_entities']['media'][0]['id']
        except:
            tweets_dict['tweet_id'] = 'na'

        tweets_dict['retweet_count'] = tweets_json['retweet_count']
        tweets_dict['favorite_count'] = tweets_json['favorite_count']
        
        tweets_list.append(tweets_dict)


# In[9]:


#converting JSON data to DataFrame
tweets_df = pd.DataFrame(tweets_list)


# ## Assessing Data
# Three datasets : twitter_archive,image_predictions,tweets_df.

# I assess the gathered data visually and programmatically 
# Detect and document the quality and tidiness issues. 

# In[10]:


# use head function to show first 10 rows in twitter_archive
twitter_archive.head(10)


# In[11]:


# use info to show summary twitter_archive dataframe
twitter_archive.info()


# issues :
# <li>There are 181 retweets (retweeted_status_id, retweeted_status_user_id, retweeted_status_timestamp)
# <li>There are 78 replies (in_reply_to_status_id, in_reply_to_user_id)
# <li>There are 2297 tweets with expanded_urls (links to images) indicating 59 tweets with missing data
# <li>The timestamp data type is string (object)
# <li>There are 4 columns for dog stages (doggo, floofer, pupper, puppo)
# <li>The columns related to retweets are not applicable for original tweets
# <li>The columns related to replies are not applicable for original tweets

# In[12]:


# null_values in twitter_archive
null_values = twitter_archive.columns[twitter_archive.isnull().any()]
twitter_archive[null_values].isnull().sum()


# In[13]:


# check duplicated tweet_id
sum(twitter_archive.duplicated('tweet_id'))


# ## dog stage

# In[14]:


twitter_archive.doggo.value_counts()


# In[15]:


twitter_archive.floofer.value_counts()


# In[16]:


twitter_archive.pupper.value_counts()


# In[17]:


twitter_archive.puppo.value_counts()


# ## dog names

# use value_counts() to  detect the issues in the name column 

# In[18]:


twitter_archive.name.value_counts().head(20)


# issue :  dog names are all capitalized, so the names in lowercase are probably not names, below array for these names

# there are 745 tweets with the dog name as "None". 

# In[19]:


lowers = twitter_archive.name.loc[twitter_archive.name.str.islower()].unique()
lowers , len(lowers)


# There are 25 name not a valid name (quilty issue)

# ## dog rating

# For rating_numerator and rating_denominator I will find the max and min values and use value_counts() to detect the issues

# In[20]:


print('Max numerator is', twitter_archive.rating_numerator.max(),'Min numerator is',twitter_archive.rating_numerator.min()
     )


# In[21]:


print('Max denominator is',twitter_archive.rating_denominator.max(),'Min denominator is',twitter_archive.rating_denominator.min()
     )


#  issues :
#  The max values are huge: 1776, 170.
#  The minimum is 0 for both.

# In[22]:


twitter_archive.rating_numerator.value_counts().sort_index(ascending = False)


# In[23]:


# there is a few rating numerator >= 15 so will choose it as a boundary
sum(twitter_archive.rating_numerator >= 15)


# There are 28 tweets with rating_numerator >= 15. The max value is 1776, which does not make sense.

# In[ ]:





# In[24]:


twitter_archive.rating_denominator.value_counts().sort_index(ascending = False)


# we just interest for tweets with rating denominator equal 10 the others not valid (quality issue)

# In[25]:


# the tweets with rating_denominator != 10 
sum(twitter_archive.rating_denominator != 10)


# There are 23 tweets with rating_denominator not 10. we only look at tweets with rating_denominator of 10.
# 
#  there are rating multiples of 10. its rating for more than 1 dog in photo.Ignoring the retweets and replies we have the following list.

# In[26]:


# display tweets that do NOT have denominator of 10,
# and are NOT replies and are NOT retweets
rating_filtr = (twitter_archive.rating_denominator != 10) &             twitter_archive.in_reply_to_status_id.isna() &             twitter_archive.retweeted_status_id.isna()

filtr_cols = ['text', 'tweet_id', 'rating_numerator', 'rating_denominator']
twitter_archive[rating_filtr][filtr_cols]


# In[27]:


print('The number of tweets with denominators that are not 10 and not replies or retweets ',sum(rating_filtr))


# In[28]:


#choos large numerator e.g 1120 to see the it's url
twitter_archive.iloc[1120].expanded_urls


# ther are many dogs in the photo

# In[29]:


# display tweets with numerator >= 15 that and denominator = 10,
# and are NOT replies and are NOT retweets
rating_filtr = (twitter_archive.rating_denominator == 10) &             (twitter_archive.rating_numerator >= 15) &             twitter_archive.in_reply_to_status_id.isna() &             twitter_archive.retweeted_status_id.isna()

filtr_cols = ['text', 'tweet_id', 'rating_numerator', 'rating_denominator']
twitter_archive[rating_filtr][filtr_cols]


# there are 5 tweets it is not replies or retweets with denominator = 10 and numerator >= 15.
# will not fixed it will dropped
# 

# In[30]:


#choos large numerator e.g 979 to see the it's url
twitter_archive.iloc[979].expanded_urls


# ## source

# In[31]:


twitter_archive.source.value_counts()


# issue (Quality):
# <li>There are 4 types of sources difficult to read, and they can be simplified 

# ### image predictions file

# In[32]:


# use head function to show first 5 rows in image_predictions
image_predictions.head(5)


# In[33]:


image_predictions.info()


# <li> There are 2075 image predictions and in the 2356 in the archive , there are 281 missing data
# <li> p1, p2 and p3 contain the dog breed predictions
# <li> p1_conf, p2_conf and p3_conf contain values for confidence 
# <li> p1_dog, p2_dog and p3_dog contain Boolean values indicating whether the prediction is true or false

# In[34]:


image_predictions.isnull().sum()


# In[35]:


image_predictions.p1_dog.value_counts()


# In[36]:


image_predictions.p2_dog.value_counts()


# In[37]:


image_predictions.p3_dog.value_counts()


# ### json data

# In[38]:


tweets_df.head(20)


# In[39]:


# use info to show summary tweets_df (json data) dataframe
tweets_df.info()


# <li> tweet_id is object should change to int for we can merge the data sets

# In[40]:


tweets_df.isnull().sum()


# <li>No missin values 

# In[41]:


tweets_df.describe()


# ## Cleaning Data

# #### twitter_archive cleaning 

# In[42]:


# make a copy for clean data
twitter_archive_clean = twitter_archive.copy()


# #### timestamp data type :

# ##### Define
# Convert timestamp from object to datetime data type.

# In[43]:


# code
twitter_archive_clean['timestamp'] = pd.to_datetime(twitter_archive_clean.timestamp)
twitter_archive_clean['timestamp'] = twitter_archive_clean.timestamp.dt.floor('s')


# In[44]:


#test
twitter_archive_clean.info()


# #### replies and retweets :

# ##### Define
# <li>All columns related to retweets or replies will be empty (interested in originals tweets).

# In[45]:


#test
twitter_archive_clean.info()


# In[46]:


#Drop all rows containing retweets, where these columns will be non-null: retweeted_status_id, retweeted_status_user_id and retweeted_status_timestamp.
sum(twitter_archive_clean.retweeted_status_id.notnull())


# In[47]:


#code
twitter_archive_clean = twitter_archive_clean[twitter_archive_clean.retweeted_status_id.isna()]


# In[48]:


#test
twitter_archive_clean


# In[49]:


# Drop all rows that are replies, those that have non-null values in these columns: in_reply_to_status_id and in_reply_to_user_id.
sum(twitter_archive_clean.in_reply_to_status_id.notnull())


# In[50]:


#code
twitter_archive_clean = twitter_archive_clean[twitter_archive_clean.in_reply_to_status_id.isna()]


# In[51]:


#test
twitter_archive_clean


# we're not interested in replies and retweets, we can drop all columns related to retweets: retweeted_status_id, retweeted_status_user_id, retweeted_status_timestamp.
# and Drop all columns related to replies: in_reply_to_status_id and in_reply_to_user_id 

# In[52]:


# drop retweeted_status_id, retweeted_status_user_id, retweeted_status_timestamp
#code
twitter_archive_clean = twitter_archive_clean.drop(['retweeted_status_id',
                                    'retweeted_status_user_id',
                                    'retweeted_status_timestamp'], axis = 1)


# In[53]:


# drop in_reply_to_status_id and in_reply_to_user_id
#code
twitter_archive_clean = twitter_archive_clean.drop(['in_reply_to_status_id',
                                    'in_reply_to_user_id'], axis = 1)


# In[54]:


# test
twitter_archive_clean.columns


# #### source :

# ##### Define
# The source column difficult to read and can be simplified by extracting and replacing with it's display string.

# In[55]:


# code
twitter_archive_clean['source'] = twitter_archive_clean['source'].str.extract('^<a.+>(.+)</a>$')


# In[56]:


#test
twitter_archive_clean.source.value_counts()


# #### rating_denominator and  rating_numerator :

# ##### Define
# <li>tweets with rating_denominator not equal to 10 not valid 
# <li>Drop the tweets with rating_denominator not equal to 10
# 

# In[57]:


#code
twitter_archive_clean = twitter_archive_clean[twitter_archive_clean.rating_denominator == 10]


# In[58]:


# test
twitter_archive_clean.rating_denominator.value_counts()


# ##### Define 
#  all rating_denominators are the same (10) this column is no longer needed

# In[59]:


#Drop the rating_denominator column not useful.
#code
twitter_archive_clean.drop(['rating_denominator'], axis = 1, inplace = True)


# ##### Define
# <li>tweets with rating_numerator >= 15 don't make sense
# <li>Drop tweets that have rating_numerator >= 15

# In[60]:


#code
twitter_archive_clean = twitter_archive_clean[twitter_archive_clean.rating_numerator < 15]


# In[61]:


#test
# display ALL numerators
twitter_archive_clean.rating_numerator.value_counts().sort_index(ascending = False)


# In[62]:


# Define: Rename the rating_numerator column to be rating
#code
twitter_archive_clean.rename(index = str, columns = {'rating_numerator': 'rating'}, inplace = True)


# In[63]:


# test
list(twitter_archive_clean)


# #### dog stages: doggo, floofer, pupper, puppo

# ##### Define
# Tidiness issu
# <li>There are 4 columns for dog stages: doggo, floofer, pupper, puppo.
# <li>I will combine in one new column (dog_stage) and Drop the 4 original dog stage columns 
#  note: I will Create a temporary column 'none' to store the None values.
# Create a new column stage to store the categories: doggo, floofer, pupper, puppo, as well as None, and select the stage from the column that contains a value.
# Drop the 4 original dog stage columns, and the temporary none column.
#     

# In[64]:


twitter_archive_clean[['doggo', 'floofer', 'pupper', 'puppo']].describe()


#  I'm going to order the dog stages by count, in increasing order: floofer, puppo, doggo and pupper.

# In[65]:


# make dummy varibles: if find value in the column set to 1 and if it is None set to zero 
#code
dummy = lambda x: 0 if x == 'None' else 1

twitter_archive_clean.doggo = twitter_archive_clean.doggo.apply(dummy)
twitter_archive_clean.floofer = twitter_archive_clean.floofer.apply(dummy)
twitter_archive_clean.pupper = twitter_archive_clean.pupper.apply(dummy)
twitter_archive_clean.puppo = twitter_archive_clean.puppo.apply(dummy)

# by adding the stage columns, we can see how many are 'none' and how many stages are set
twitter_archive_clean['none'] = twitter_archive_clean['doggo'] + twitter_archive_clean['floofer'] +                         twitter_archive_clean['pupper'] + twitter_archive_clean['puppo']


# In[66]:


# test
twitter_archive_clean['none'].value_counts()


# there are 1740 None and 324 value in none column , 2 indacites there are 11 tweets have more than dog stage

# In[67]:


# if there are NO stages specified then set 'None' to 1
none_values = lambda x: 1 if x == 0 else 0

# reset values in 'none' 
twitter_archive_clean['none'] = twitter_archive_clean['none'].apply(none_values)

# Order the stages in increasing count order: floofer, puppo, doggo and pupper
# set the choice order for dog stage based on count order
dog_stage = ['floofer', 'puppo', 'doggo', 'pupper', 'none']

# set the conditions for selecting the dog stage based on count order
conditions = [
    (twitter_archive_clean[dog_stage[0]] == 1),
    (twitter_archive_clean[dog_stage[1]] == 1),
    (twitter_archive_clean[dog_stage[2]] == 1),
    (twitter_archive_clean[dog_stage[3]] == 1),
    (twitter_archive_clean[dog_stage[4]] == 1)]

# select the dog stage based on the first successful condition; stage[4] is 'None'
twitter_archive_clean['dog_stage'] = np.select(conditions, dog_stage, default = dog_stage[4])

# now we can drop the original 4 dog stage columns, AND the temporary 'None'
twitter_archive_clean.drop(dog_stage, axis = 1, inplace = True)

# set the 'stage' column data type to category
twitter_archive_clean['dog_stage'] = twitter_archive_clean.dog_stage.astype('category')


# In[68]:


# test
twitter_archive_clean.info()


# In[69]:


twitter_archive_clean.dog_stage.value_counts()


#  #### image_predictions cleaning

# In[70]:


# make a copy for cleaning
image_clean = image_predictions.copy()


# In[71]:


# drop the false prediction
#code
image_clean.drop(image_clean[image_clean.p1_dog == False].index, inplace=True)


# In[72]:


#test
image_clean.p1_dog.value_counts()


# In[73]:


# rename p1 to dog_breed_prediction and p1_conf to prediction_confidence
# code
col_names = {'p1':'dog_breed_prediction', 'p1_conf':'prediction_confidence'}

image_clean.rename(columns= col_names, inplace=True)


# In[74]:


# drop the rest columns
col_drop = ['jpg_url', 'p1_dog', 'p2', 'p2_conf', 'p2_dog', 'p3', 'p2_conf', 'p3_dog', 'p3_conf']

image_clean.drop(col_drop, inplace=True, axis=1)


# In[75]:


# Test
list(image_clean.columns)


# #### json data claening

# In[76]:


tweets_df_clean = tweets_df.copy()


# In[77]:


# Define: Finding non-numeric values for "tweet_id"
non_num = []

for i in range(0, len(tweets_df_clean.tweet_id)):
    if type(tweets_df_clean.tweet_id[i]) != int:
        non_num.append(i)
      


# In[78]:


print (len(non_num))


# In[79]:


tweets_df_clean.shape


# In[80]:


for i in non_num:
    tweets_df_clean.drop(tweets_df_clean[tweets_df_clean.index == i].index, inplace=True)


# In[81]:


tweets_df_clean.shape


# In[82]:


# Reset index
tweets_df_clean = tweets_df_clean.reset_index()
del tweets_df_clean['index']


# In[83]:


# Define: change the 'tweet_id' data type to int
#code
tweets_df_clean['tweet_id']=tweets_df_clean['tweet_id'].astype(int)
tweets_df_clean['tweet_id']=tweets_df_clean['tweet_id'].astype(np.int64)


# In[84]:


#test
tweets_df_clean.info()


# ## Store Data

# In[85]:


# Merge the tweets_df_clean to the twitter_archive_clean, joining on tweet_id.
twitter_archive_clean = pd.merge(twitter_archive_clean, tweets_df_clean, 
                         on = 'tweet_id', how = 'left')


# In[86]:


# then Merge image_clean to the twitter_archive_clean, joining on tweet_id
twitter_archive_clean = pd.merge(twitter_archive_clean, image_clean, 
                         on = 'tweet_id', how = 'left')


# In[87]:


# save to csv file
twitter_archive_clean.to_csv('clean_twitter_archive_data.csv')


# ## Analysis

# In[99]:


we_rate_dogs.shape


# In[88]:


# make a copy of the archive for analysis
we_rate_dogs = twitter_archive_clean.copy()
we_rate_dogs.info()


# #### Question:
# #### what is the Percentage of tweets with rating of 10 and up ?

# In[90]:


# Percentage of tweets with rating of 10 and up
high_rating = sum(we_rate_dogs.rating >= 10)
high_perc = round(high_rating * 100 / we_rate_dogs.shape[0])
print("Number of tweets with rating up to 10:  {}".format(high_rating))
print("Percentage of tweets:                {}%".format(round(high_perc, 3)))


# In[89]:


we_rate_dogs['rating'].plot(kind = 'hist', bins = 15)

plt.xlim(0, 15)
plt.ylabel('Number of Tweets', fontsize = 14)
plt.xlabel('Rating', fontsize = 14)
plt.title('Distribution of Ratings', fontsize = 16)
plt.show();


# The distribution of ratings is very skewed to the left.
# the IQR is from 10 to 12
# From statistics  we see that 80% of all ratings are up 10 

# #### Question: 
# #### What is the most common stage?

# In[91]:


we_rate_dogs.dog_stage.value_counts()


# In[92]:


#chart common stage ( ignore none)
dog_type = ['pupper', 'doggo', 'puppo', 'floofer']
dog_counts = [220, 81, 24, 10]

fig,ax = plt.subplots(figsize = (12,6))
ax.bar(dog_type, dog_counts, width = 0.8)
ax.set_ylabel('Dog Count')
ax.set_xlabel('Dog Stage')
plt.title("Most Common Dog Stage")
plt.show()


# The chart show the most popular dog type is a "pupper" it is up than 200, then doggo , the last common stage is floofer 

# #### Question:
# #### What are the 10 most dog breeds tweeted about? (Excluding the category 'none'.)

# In[93]:


none_count = sum(we_rate_dogs.dog_breed_prediction == 'none')

print("Number of tweets with 'none' predicted breed:",none_count)


# In[94]:


dist_breeds = len(we_rate_dogs.dog_breed_prediction.unique())

print("Number of distinct breeds:" , dist_breeds) 


# In[95]:


top10_breeds_count = we_rate_dogs[we_rate_dogs.dog_breed_prediction != 'none'].dog_breed_prediction.value_counts().head(10)
print("Breed and number of tweets:")
print(top10_breeds_count)


# In[96]:


plt.barh(top10_breeds_count.index, top10_breeds_count)

plt.xlabel('Number of Tweets', fontsize = 14)
# plt.ylabel('Dog Breed', fontsize = 14)
plt.title('Top 10 Dog Breeds by Tweet Count', fontsize = 16)
plt.gca().invert_yaxis()
plt.show();


# The most dog breed tweeted about is golden_retriever 
# The bar chart shows that the most ten common dog breed that is tweeted about, the Golden Retriever was the most with almost 140 tweets, Labrador Retrievers, Pembrokes and Chihuahuas are fairly close together in 2nd, 3rd, and 4th place.

# In[ ]:




