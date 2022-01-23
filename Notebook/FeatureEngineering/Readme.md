# Features Engineering

<p align="center">
  <img width="1000" height="500" src="https://www.ismiletechnologies.com/wp-content/uploads/2021/09/Feature-Engineering-in-machine-learning.png">
</p>

## Contents:
 
 * Introduction
 * Fixing The Existing Tensorflow Dataset (movie_lens/100k-dataset)
 * Adding five more features to the original Dataset: 'cast', 'director', 'cast_size', 'crew_size', 'imdb_id', 'release_date' and 'movie_lens_movie_id'
 * Convert Pandas Dataset to Tensorflow Dataset


## 1. Introduction: 

Objective: to improve our cleaned Dataframe which was orginally from the Tensorflow Dataset: movie_lens/100k-ratings:

- Fixing "movie_genres": formatting to a list.

- Checking the "user_occupation_label": making sure that the classificiations match the documentation.

- Adding 5 more features to the original Dataset: 'cast', 'director', 'cast_size', 'crew_size', 'imdb_id', 'release_date' and movie_lens_movie_id, from:

    - movies_metadata.csv
    - credits.csv
    
- Removing all special characters or letter accents from certain columns.

- Converting the Pandas dataframe to tensforflow datasets.

## 2. Fixing The Existing Tensorflow Dataset (movie_lens/100k-dataset)

 <p align="center">
  <img width="1000" height="500" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/classes.png">
</p>

The data classification for the "user_occupation_label" looks as expected


## 3. Adding five more features to the original Dataset: 'cast', 'director', 'cast_size', 'crew_size', 'imdb_id', 'release_date' and 'movie_lens_movie_id'

We will ne using those two datasets downloaded from the MovieLens website

- [movies_metadata.csv](https://grouplens.org/datasets/movielens/)
- [credits.csv](https://grouplens.org/datasets/movielens/)

1. Crew: From the crew, we will only pick the director as our main feature.

2. Cast: Choosing Cast is a little more tricky. Lesser known actors and minor roles do not really affect people's opinion of a movie. Therefore, we must only select the major characters and their respective actors. Arbitrarily we will choose the top 4 actors that appear in the credits list.

## 4. Convert Pandas Dataset to Tensorflow Dataset

We need to convert our cleaned pandas dataframe to a tensorflow dataset that TFRS can read:

- From 'cast' features, dropping all secondary casting and keep only the star of the movie and let's call the feature "star"

- Only keeping the important columns.

- Changing the data types of the important features to fit the Tensorflow-Recommender TFRS Library.

    - tfds currently does not support float64 so we'll be using int64 or float32 depending on the data.
    - We'll wrap the pandas dataframe into a tf.data.Dataset object using tf.data.Dataset.from_tensor_slices 
    (To check other options - [here] ('https://www.srijan.net/resources/blog/building-a-high-performance-data-pipeline-with-tensorflow#gs.f33srf')



