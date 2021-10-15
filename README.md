# Lyric Generation - Red Hot Chili Peppers
Author: Katarina Salcedo

# Motivation
On average it take anywhere from 3-30 months to complete an album. My goal is to come up with a solution to streamline this process and assist with the creation of new songs/lyrics that can be used in upcoming albums. Since each artist is unique - we will focus on a single artist, in this case the Red Hot Chili Peppers (RHCP), and use deep learning to generate new lyrics that match the style and voice of that particular artist. We will also use machine learning to predict the popularity of the song(s) given the lyrics.

# Data
A dataset was synthesized using Spotify's API along with Genius Lyric API. It contained 118 songs from all 10 of the RHCP's albums and had information on audio features, popularity and song lyrics. The audio features include: danceability, energy, key, loudness, speechiness, acousticness, liveness, instrumentalness, valence and tempo. 

# Methodology
The methodolody is split between the deep learning and machine learning components of this project because the dataset is manipulated a differently for each model. 

## Language Model
To generate song lyrics, we'll develop a statistical language model that utilizes deep learning. This language model will be able to predict the probabiity of the next word in the sequence given an input sequence. To prepare the data, sentences were tokenized using Keras Tokenizer() and padded sequences were created for each sentence. For training, we will use an Embedding Layer to learn the representation of words, and a Long Short-Term Memory (LSTM) recurrent neural network to learn to predict words based on their context. The LSTM network has feedback connections so it is capable of learning order dependence in sequence prediction which makes it a good model for text generation. The model will learn the sentence/line structure and how each word is being arranged to build new lyrics. The model currently has an accuracy of 0.495 and loss of 2.052. This means that the model is 49% accurate at predicting the next word. This accuracy is pretty good since we are not aiming for 100% accuracy (a model that memorized the text), but rather a model that captures the essence of the text. Below is a graph summarizing the model's training.

![Screen Shot 2021-10-14 at 11 16 13 PM](https://user-images.githubusercontent.com/81720110/137441015-eb26c677-6115-4dd8-9ed2-060345c06b00.png)
![Screen Shot 2021-10-14 at 11 16 22 PM](https://user-images.githubusercontent.com/81720110/137441021-5414f908-af73-4ae8-995f-b269b3e6d526.png)


## Regression Model
For the regression model, each word was tokenized and stop words were removed. TFIDF Vectorizer was then used to create a matrix of Tf-IDF features that will act as the independent variable (x) in the model. The dependent variables that we want to predict from the lyric data include the audio features along with popularity. Four baseline regression models were tested - Linear regression, Linear SVR, Lasso Lars and SGDRegressor. Of these four the SGDRegressor preformed the best and was tuned using Gridsearch. Below are the results of the model. 

![Screen Shot 2021-10-14 at 11 07 38 PM](https://user-images.githubusercontent.com/81720110/137440360-05366364-bdee-4d8e-876c-5d4e9e6853e5.png)
![Screen Shot 2021-10-14 at 11 07 57 PM](https://user-images.githubusercontent.com/81720110/137440374-86af3940-05cf-483b-9972-e9ecb491c137.png)


# Results 
Below is a list of generated lyrics that can be used to create a song:

![Screen Shot 2021-10-14 at 11 18 44 PM](https://user-images.githubusercontent.com/81720110/137441248-cf646f83-c5b5-4147-be70-aae6868db85e.png)

Based on these lyrics, here are the predicted audio features and popularity:

![Screen Shot 2021-10-14 at 11 20 16 PM](https://user-images.githubusercontent.com/81720110/137441378-a6ad0402-d2e9-442e-9a55-59a569a46010.png)

# Conclusions 
Currently, the generated lyrics will produce a song with a popularity score of 54. For reference, the highest score the Red Hot Chili Peppers have recieved is 73, with a majority of their scores around the 50-65 range. 

# Next Steps
Future work would include expanding the dataset to include all top songs from a particular genre. Increasing the size of the dataset would also improve upon the modeling and allow for the creation of a more general text preditor. To improve upon the language model, using pretrained networks could produce a model that is better at predicting longer sequences. 
