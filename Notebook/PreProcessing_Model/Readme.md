# Modeling

<p align="center">
  <img width="600" height="300" src="https://www.tensorflow.org/site-assets/images/project-logos/tensorflow-recommenders-logo-social.png">
</p>

## Contents:
 
 * Introduction
 
 * Sourcing and Loading
   * Importing relevant libraries
 
 * Dataset
   * Preparing the datasets

 * Features Importance Using Deep & Cross Network (DCN-V2)
   * Deep and cross network (DCN) came out of Google Research, and is designed to learn explicit and bounded-degree cross features effectively
   * Feature Cross
   * Cross Network
   * Deep & Cross Network
   * Model Structure
   * Model construction
   * Model Training
   * DCN (stacked)
   * Low-rank DCN
   * DNN (Cross Layer = False)
   
   
 * The Two-Tower and Ranking Models
   * Case 1: Baseline
       * A Multi-Task Model
           * Rating-specialized model
		   * Retrieval-specialized model
		   * Joint model: Baseline
            * Summary of the Baseline Joint Model
           
   * Case 2: Tuned Joint Model
        * Implementing a Retrieval Model
		* A Multi-Task Model
		* Joint model
        
 * Summary of all models' metrics


## 1. Introduction: 

For this final notebook, we will be dealing with:

- The Feature importance using Deep and Cross Network (DCN-v2)

- Training multiple TensorFlow Recommenders.

- Applying hyperparameters tuning where applicable to ensure every algorithm will result in the best prediction possible.

- Finally, evaluating these Models.

## 2. Dataset

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

**[movie_lens/100k-movies](https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens100k-movies):**

 * Config description: This dataset contains data of approximately 1,682 movies rated in the 100k dataset.
 * Download size: 4.70 MiB
 * Dataset size: 150.35 KiB
 * Auto-cached ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)): Yes
 * Features:
```
FeaturesDict({
              'movie_genres': Sequence(ClassLabel(shape=(), dtype=tf.int64, num_classes=21)),
              'movie_id': tf.string,
              'movie_title': tf.string,
            })

```
## 3. Features Importance Using Deep & Cross Network (DCN-V2)

<p align="center">
  <img width="600" height="300" src="https://camo.githubusercontent.com/ba6bec17d331afbddc49fe9f0b88caa133b58979e78c5bafe9f10091cd9c7404/687474703a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315838716f4d7449594b4a7a3479425969667666773451704177726a723730655f">
</p>


**Deep and cross network (DCN) came out of Google Research, and is designed to learn explicit and bounded-degree cross features effectively**

- Large and sparse feature space is extremely hard to train.
- Oftentimes, we needed to do a lot of manual feature engineering, including designing cross features, which is very challenging and less effective.
- Whilst possible to use additional neural networks under such circumstances, it's not the most efficient approach. DCN is specifically designed to tackle all of the above challenges.

**Feature Cross**

Let's say we're building a recommender system to sell a blender to customers. Then our customers' past purchase history, such as purchased apples and purchased recipes books, or geographic features are single features. If one has purchased both apples and recipes books, then this customer will be more likely to click on the recommended blender. The combination of purchased apples and the purchased recipes books is referred to as feature cross, which provides additional interaction information beyond the individual features.

**Cross Network**

In real world recommendation systems, we often have large and sparse feature space. So, identifying effective feature processes in this setting would often require manual feature engineering or exhaustive search, which is highly inefficient. To tackle this issue, Google Research team has proposed DCN. It starts with an input layer, typically an embedded layer, followed by a cross network containing multiple cross layers that models explicitly feature interactions, and then combines with a deep network that models implicit feature interactions. The deep network is just a traditional multilayer construction. But the core of DCN is really the cross network. It explicitly applies feature crossing at each layer. And the highest polynomial degree increases with layer depth.

**Deep & Cross Network**

There are a couple of ways to combine the cross network and the deep network:

- Stack the deep network on top of the cross network.
- Place deep & cross networks in parallel.

**Model Structure**

We first train a DCN model with a stacked structure, that is, the inputs are fed to a cross network followed by a deep network.

**Model construction**

- The model architecture we will be building starts with an embedded layer, which is fed into a cross network followed by a deep network. 
- The embedded dimension is set to 32 for all the features. 
- Please note that we could also have used different embedded sizes for different features.

**Model Training**

We set some hyper-parameters for the models. 

- Note that these hyper-parameters are set globally for all the models for demonstration purpose. 

- If you want to obtain the best performance for each model, or to conduct a fair comparison amongst models, I would suggest you to fine-tune the hyper-parameters. 

- Remember that the model architecture and optimization schemes are intertwined.

**DCN (stacked)**

<p align="center">
  <img width="600" height="300" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/graph_featureimportance.png">
</p>

From the above graph, we can visualize the weights from the cross network and see if it has successfully learned the important feature process. 

As shown above, fro instance, the feature cross of user ID and movie ID are of great importance.

**Low-rank DCN**

To reduce the training and running cost, we leverage low-rank techniques to approximate the DCN weight matrices. 

- The rank is passed in through the projection_dim argument: a smaller projection_dim results in a lower cost. 

- Note that projection_dim needs to be smaller than the (input size)/2 to reduce the cost. In practice, we've observed that using a low-rank DCN with a rank of (input size)/4 consistently preserved the accuracy of a full-rank DCN.

**DNN (Cross Layer = False)**

DCN            RMSE mean: 0.9811, stdv: 0.0241
DCN (low-rank) RMSE mean: 0.9666, stdv: 0.0128
DNN            RMSE mean: 0.9531, stdv: 0.0115

## 4. Multi-Task Model - Joint Model

**The Two-Tower and Ranking Models**

<p align="center">
  <img width="600" height="300" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/twotower.png">
</p>


**Real-world recommender systems are often composed of two stages:**

- *The retrieval stage* (Selects recommendation candidates): is responsible for selecting an initial set of hundreds of candidates from all possible candidates. 
    - The main objective of this model is to efficiently weed out all candidates that the user is not interested in. Because the retrieval model may be dealing with millions of candidates, it has to be computationally efficient.

- *The ranking stage* (Selects the best candidates and rank them): takes the outputs of the retrieval model and fine-tunes them to select the best possible handful of recommendations. 
    - Its task is to narrow down the set of items the user may be interested in to a shortlist of likely candidates.

**Retrieval models are often composed of two sub-models:**

The retrieval model embeds user ID's and movie ID's of rated movies into embedded layers of the same dimension:

- A query model computing the query representation (normally a fixed-dimensionality embedding vector) using query features.

- A candidate model computing the candidate representation (an equally-sized vector) using the candidate features.

- As shown below, the two models are multiplied to create a query-candidate affinity scores for each rating during training. If the affinity score for the rating is higher than  other candidates, then we can consider the model to be reasonable.

**Embedded layer Magic**

As discussed above, we might think of the embedded layers as just a way of encoding a way of forcing the categorical data into some sort of a standard format that can be easily fed into a neural network and usually that's how it's used but embedded layers are more than that! 

The way they're working under the hood is every unique id is being mapped to a vector of n dimensions, but it's going to be like a vector of 32 floating point values and we can think of this as a position in a 32-dimensional space that represents the similarity between one user id and another or between one movie id and another so by using embeddied layers in this way we're getting around that whole problem of data sparsity and sparse vectors and at the same time, we're getting a measure of similarity  so it's a very simple way of getting recommendation candidates.

The outputs of the two models are then multiplied together to give a query-candidate affinity score, with higher scores expressing a better match between the candidate and the query.

In this Model, we built and trained such a two-tower model using the Movielens dataset (100k Dataset):

- Getting our data and splitting it into a training and test set.
- Implementing a retrieval model.
- Fitting and evaluating it.

**The Ranking**

The ranking stage takes the outputs of the retrieval model and fine-tunes them to select the best possible handful of recommendations. 
- Its task is to narrow down the set of items the user may be interested in to a shortlist of likely candidates.

**Case 1: Baseline**

There are two critical parts to multi-task recommenders:

- They optimize for two or more objectives, and so have two or more losses.
- They share variables between the tasks, allowing for transfer learning.

Now, let's define our models as before, but instead of having a single task, we will have two tasks: one that predicts ratings, and one that predicts movie watches.

**Putting it together**

- We put it all together in a model class.

- The new component here is that - since we have two tasks and two losses - we need to decide on how important each loss is. We can do this by giving each of the losses a weight, and treating these weights as hyperparameters. 
    - If we assign a large loss weight to the rating task, our model is going to focus on predicting ratings (but still use some information from the retrieval task); if we assign a large loss weight to the retrieval task, it will focus on retrieval instead.

#### Rating-specialized model

**Depending on the weights we assign, the model will encode a different balance of the tasks. Let's start with a model that only considers ratings**

<p align="center">
  <img width="600" height="300" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/lossepochs.png">
</p>

As the model trains, the loss is falling and a set of top-k retrieval metrics is updated. 
- These tell us whether the true positive is in the top-k retrieved items from the entire candidate set. 
- For example, a top-5 categorical accuracy metric of 0.2 would tell us that, on average, the true positive is in the top 5 retrieved items 20% of the time.

11/11 [==============================] - 2s 110ms/step - root_mean_squared_error: 1.0711 - factorized_top_k/top_1_categorical_accuracy: 0.0723 - factorized_top_k/top_5_categorical_accuracy: 0.1022 - factorized_top_k/top_10_categorical_accuracy: 0.1153 - factorized_top_k/top_50_categorical_accuracy: 0.1852 - factorized_top_k/top_100_categorical_accuracy: 0.2359 - loss: 1.1437 - regularization_loss: 0.0000e+00 - total_loss: 1.1437
Retrieval top-100 accuracy: 0.236.
Ranking RMSE: 1.071.
We get a good RMSE but with poor prediction

### Retrieval-specialized model
**Let's now try a model that focuses on retrieval only.**

11/11 [==============================] - 2s 134ms/step - root_mean_squared_error: 3.6912 - factorized_top_k/top_1_categorical_accuracy: 3.1586e-04 - factorized_top_k/top_5_categorical_accuracy: 6.7683e-04 - factorized_top_k/top_10_categorical_accuracy: 0.0013 - factorized_top_k/top_50_categorical_accuracy: 0.0229 - factorized_top_k/top_100_categorical_accuracy: 0.0578 - loss: 15094.9468 - regularization_loss: 0.0000e+00 - total_loss: 15094.9468
Retrieval top-100 accuracy: 0.058.
Ranking RMSE: 3.691.
We get a less impressive RMSE coupled with a poor prediction too

### Joint model: Baseline

**Let's now train a model that assigns positive weights to both tasks.**

<p align="center">
  <img width="600" height="300" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/metricsintermediary.png">
</p>

**The joint model seems to provide an overall better prediction than the other independent models**
**What can we improve upon the joint model?**

- Adding more features
- Optimize Embedding
- embedding_dimension
- epochs= 8 to 16
- Learning rate

#### Case 2: Tuned Joint Model

As before, this task's goal is to predict which movies the user will or will not watch.

Those folowwing classes were re-configured to accomodate the new embedded design due to new features, having deeper neural networks and adding regularization to help overfitting:

- class UserModel
- class QueryModel
- class MovieModel
- class MovieModel
- class CandidateModel
- class MovielensModel

Let's now train a model that assigns positive weights to both tasks.

<p align="center">
  <img width="600" height="300" src="https://raw.githubusercontent.com/mohamedziane/Movie-Recommendation-System-TensorflowRS/main/images/modesmetrics.png">
</p>

## 5. Future Work

 * Further optimizing existing Tensorflow Recommenders Classes to accommodate extra features.
 * Additional fine-tuning  needed to enhance retrieval prediction and reduce RMSE Ranking from the Joint Baseline Model (Multi-Task)
 * Building an application