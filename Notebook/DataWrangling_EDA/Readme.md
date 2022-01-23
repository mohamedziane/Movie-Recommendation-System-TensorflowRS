# Data Wrangling & Exploratory Data Analysis EDA

<p align="center">
  <img width="800" height="500" src="INSERT">
</p>

## Contents:
 
 * Introduction
 * Dataset
 * Sourcing and Loading
   * Import relevant libraries
   * Converting Tensforflow Dataset to Pandas DataFrame
 * Data Wrangling 
 * Exploratory Data Analysis (EDA)
   * What is the most preferrable day to rate/watch movies?
   * Who, among men and women, watches/rates more movies?
   * What age group watches more movies?
   * What kind of occupation do users have that watch/rate movies the most?
   * Let's have more insights between male and female users
   * What are the most rated movies?
   * What are the most liked Movies?
   * Which year the users were interested the most to rate/watch movies?
   * What are the worst movies per rating?? ***Using worldcloud Package***
   * Is there any relation between the users rate and their geographical location? 
   * Whats the most popular Genre in our dataset?

## 1. Introduction: 


On the one hand, the **The Data wrangling step** focuses on collecting or converting the data, organizing it, and making sure it's well defined. For our project we will be using the ***movie_lens/100k dataset*** from TensorFlow  because it's a unique dataset with plenty of Metadata that's needed for this project. We'll focus in particular on:


 * Cleaning NANs (If any), duplicate values (If any), wrong values and removing insignificant columns.
 * Removing any special characters.
 * Renaming some Column labels.
 * Correcting some datatypes.
 
On the other hand, the **Exploratory Data Analysis EDA Step** will focus on: 
 
 * Getting familiar with the features in our dataset.
 * Understanding the core characteristics of our cleaned dataset.
 * Exploring the data relationships of all the features and understanding how the features compare to the response variable.
 * Further analyzing interesting plots to deepen our understanding of the data.

## 2. Dataset: 

**Movie Lens** contains a set of movie ratings from the MovieLens website, a movie recommendation service. This dataset was collected and maintained by [GroupLens](https://grouplens.org/) , a research group at the University of Minnesota. There are 5 versions that include: "25m", "latest-small", "100k", "1m", "20m". In all of the datasets, the movies and ratings data are joined on "movieId". The 25m dataset, latest-small dataset, and 20m dataset contain only the movie and rating data. The 1m dataset and 100k dataset contain demographic data in addition to the movie and rating data.


**[movie_lens/100k-ratings](https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens100k-ratings):**
 * Config description: This dataset contains 100,000 anonymous ratings of approximately 1,682 movies made by 943 MovieLens users who joined MovieLens. Ratings are in whole-star increments. This dataset contains demographic data of users in addition to data on movies and ratings.
 
 * This dataset is the second largest dataset that includes demographic data from movie_lens.
 * "user_gender": gender of the user who made the rating; a true value corresponds to male.
 * "bucketized_user_age": bucketized age values of the user who made the rating, the values and the corresponding ranges are:
   * 1: "Under 18"
   * 18: "18-24"
   * 25: "25-34"
   * 35: "35-44"
   * 45: "45-49"
   * 50: "50-55"
   * 56: "56+"
 * "movie_genres": The Genres of the movies are classified into 21 different classes as below:
   * 0: Action
   * 1: Adventure
   * 2: Animation
   * 3: Children
   * 4: Comedy
   * 5: Crime
   * 6: Documentary
   * 7: Drama
   * 8: Fantasy
   * 9: Film-Noir
   * 10: Horror
   * 11: IMAX
   * 12: Musical
   * 13: Mystery
   * 14: Romance
   * 15: Sci-Fi
   * 16: Thriller
   * 17: Unknown
   * 18: War
   * 19: Western
   * 20: no genres listed
   
 * "user_occupation_label": the occupation of the user who made the rating represented by an integer-encoded label; labels are preprocessed to be consistent across different versions
 * "user_occupation_text": the occupation of the user who made the rating in the original string; different versions can have different set of raw text labels
 * "user_zip_code": the zip code of the user who made the rating.
 * Download size: 4.70 MiB
 * Dataset size: 32.41 MiB
 * Auto-cached ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)): No
 * Features:
 ```
 FeaturesDict({
               'bucketized_user_age': tf.float32,
               'movie_genres': Sequence(ClassLabel(shape=(), dtype=tf.int64, num_classes=21)),
               'movie_id': tf.string,
               'movie_title': tf.string,
               'raw_user_age': tf.float32,
               'timestamp': tf.int64,
               'user_gender': tf.bool,
               'user_id': tf.string,
               'user_occupation_label': ClassLabel(shape=(), dtype=tf.int64, num_classes=22),
               'user_occupation_text': tf.string,
               'user_rating': tf.float32,
               'user_zip_code': tf.string,
              })
 ```
**Example:**

|bucketized_user_age	|movie_genres|	movie_id|	movie_title|	raw_user_age|	timestamp|	user_gender|	user_id	|user_occupation_label|	user_occupation_text	|user_rating	|user_zip_code|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|45.0	|7 (Drama)|b'357'	|b"One Flew Over the Cuckoo's Nest (1975)"	|46.0	|879024327	|True	|b'138'	|4 (doctor/health care)	|b'doctor'	|4.0|	b'53211'|

## 3. Data Wrangling Objectives

 * Changing the user_gender from booleans "Female" or "Male" to the following association: True:"Male", False:"Female"
 * Removing the symbols: (b), (') and (").
 * Dropping the following columns: "user_occupation_label" and "movie_genres".
 * Changong "timestamp" which is in the unix epoch (units of seconds) to a datetime64 type.
 * Fixing any wrong values in "user_zip_code" (removing any zipcode >5 characters & zip codes made out of letters)


## 4. EDA

### What to expect in the EDA steps?

 * To get familiar with the features in our dataset.
 * Generally understand the core characteristics of our cleaned dataset.
 * Explore the data relationships of all the features and understand how the features compare to the response variable.
 * We will think about interesting figures and all the plots that can be created to help deepen our understanding of the data.
 * We will be creating one feature that give us the year when the movie was released and will call it "movie_year_release".

 **(1) What is the most preferrable day to rate/watch movies?**

 <p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/1.png">
</p>

Even though mid-week seems to slightly stand out (Wednesday), overall the distribution is almost equal and the day of the week does not appear to be a factor on to when users watches/rates movies

 **(2) Who, among men and women, watches/rates more movies?**

 <p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/2.png">
</p>

Looks like male users are rating more movies than females

 **(3) What age group watches more movies?**

 <p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/3.png">
</p>

Group ranking by age for watching/rating movies

* First: 25 (25-34)
* Second: 18 (18-24)
* Third: 35 (35-44)

 **(4) What kind of occupation do users have that watch/rate movies the most?**

 <p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/4.png">
</p>

The 18-24 age group for students lead the way

 **(5) Let's have more insights between male and female users**

 <p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/5.png">
</p>

On a rating scale from 1 to 5, both male and female give more "4" ratings

 **(6) What are the most rated movies?**

 <p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/6.png">
</p>

<p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/6_2.png">
</p>

<p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/6_3.png">
</p>

<p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/6_4.png">
</p>

<p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/6_5.png">
</p>

<p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/6_6.png">
</p>


 **(7) What are the most liked Movies?**

 <p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/7_1.png">
</p>

<p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/7_2.png">
</p>

<p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/7_3.png">
</p>

<p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/7_4.png">
</p>

<p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/7_5.png">
</p>

<p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/7_6.png">
</p>


 **(8) What are the worst movies per rating?**

 <p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/8.png">
</p>


 **(9) Is there any relation between the users rate and their geographical location?** 

 <p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/9.png">
</p>

California and Texas lead the way. Are the students the group who's also causing such high rates?
Indeed, students are the group driving the rating in California

 **(10) Whats the most popular Genre in our dataset?**

 <p align="center">
  <img width="800" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/10.png">
</p>

We can infer from the above graph that the 3 top most popular Genres are Adventure, Comedy and Drama:

The Genres of the movies are classified into 21 different classes as below:
- 0: Action
- 1: Adventure
- 2: Animation
- 3: Children
- 4: Comedy
- 5: Crime
- 6: Documentary
- 7: Drama
- 8: Fantasy
- 9: Film-Noir
- 10: Horror
- 11: IMAX
- 12: Musical
- 13: Mystery
- 14: Romance
- 15: Sci-Fi
- 16: Thriller
- 17: Unknown
- 18: War
- 19: Western
- 20: no genres listed

