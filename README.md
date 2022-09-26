# Text-Classification---LSTM

### Natural Language Processing

NLP is a branch of Artificial Intelligence which deal with bridging the machines understanding humans in their Natural Language. Natural Language can be in form of text or sound, which are used for humans to communicate each other. NLP can enable humans to communicate to machines in a natural way.

### Text Classification

is a process involved in Sentiment Analysis. It is classification of peoples opinion or expressions into different sentiments. Sentiments include Positive, Neutral, and Negative, Review Ratings and Happy, Sad. Sentiment Analysis can be done on different consumer centered industries to analyse people's opinion on a particular product or subject.

### Sentiment Classification

is a perfect problem in NLP for getting started in it. You can really learn a lot of concepts and techniques to master through doing project. Kaggle is a great place to learn and contribute your own ideas and creations. I learnt lot of things from other, now it's my turn to make document my project.


![Screenshot 2022-09-26 072616](https://user-images.githubusercontent.com/56751947/192302137-9660b79c-8c0d-42f1-b09a-864f3737851a.jpg)


## DATASET
The dataset is taken from Kaggle and can be found [here](https://www.kaggle.com/datasets/kazanova/sentiment140)

It consists of  1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

It contains the following 6 fields:

- **target:** the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

- **ids:** The id of the tweet ( 2087)

- **date:** the date of the tweet (Sat May 16 23:58:44 UTC 2009)

- **flag:** The query (lyx). If there is no query, then this value is NO_QUERY.

- **user:** the user that tweeted (robotickilldozr)

- **text:** the text of the tweet (Lyx is cool)

## DATA VISUALIZATON

POSITIVE WORDS

![Screenshot 2022-09-26 071304](https://user-images.githubusercontent.com/56751947/192299182-a0f2473d-6848-4e64-8566-f783f75949af.jpg)

NEGATIVE WORDS

![Screenshot 2022-09-26 071626](https://user-images.githubusercontent.com/56751947/192299996-c6450d4f-4c69-4101-9b20-23af76d8fc56.jpg)

## TOKENIZATION

Given a character sequence and a defined document unit, tokenization is the task of chopping it up into pieces, called tokens , perhaps at the same time throwing away certain characters, such as punctuation. The process is called Tokenization.

![Screenshot 2022-09-26 072722](https://user-images.githubusercontent.com/56751947/192302407-048a1b3c-ed5e-467c-923e-0fa664f27e7f.jpg)


`tokenizer` create tokens for every word in the data corpus and map them to a index using dictionary.

`word_index` contains the index for each word

`vocab_size` represents the total number of word in the data corpus

## WORD EMBEDDING

In Language Model, words are represented in a way to intend more meaning and for learning the patterns and contextual meaning behind it.

Word Embedding is one of the popular representation of document vocabulary.It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc.

Basically, it's a feature vector representation of words which are used for other natural language processing applications.

We could train the embedding ourselves but that would take a while to train and it wouldn't be effective. So going in the path of Computer Vision, here we use Transfer Learning. We download the pre-trained embedding and use it in our model.

The pretrained Word Embedding like GloVe & Word2Vec gives more insights for a word which can be used for classification.

In this notebook, I use GloVe Embedding from Stanford AI which can be found [here](https://nlp.stanford.edu/projects/glove/)


## LSTM - SEQUENCE MODELS

Some words are predominantly feature in both Positive and Negative tweets. This could be a problem if we are using a Machine Learning model like Naive Bayes, SVD, etc.. That's why we use Sequence Models.

Reccurent Neural Networks can handle a seqence of data and learn a pattern of input seqence to give either sequence or scalar value as output. In our case, the Neural Network outputs a scalar value prediction.

![Screenshot 2022-09-26 072808](https://user-images.githubusercontent.com/56751947/192302551-8ecd68e2-0409-49d8-951d-3a55a22ccf53.jpg)

For model architecture, we use

1) **Embedding Layer** - Generates Embedding Vector for each input sequence.

2) **Conv1D Layer** - Its using to convolve data into smaller feature vectors.

3) **LSTM** - Long Short Term Memory, its a variant of RNN which has memory state cell to learn the context of words which are at further along the text to carry contextual meaning rather than just neighbouring words as in case of RNN.

4) **Dense** - Fully Connected Layers for classification


## MODEL ACCURACY 

![Screenshot 2022-09-26 072241](https://user-images.githubusercontent.com/56751947/192301275-acec66f7-97af-4ff9-b3f9-6ca374d83219.jpg)


### CONFUSION MATRIX

![Screenshot 2022-09-26 072332](https://user-images.githubusercontent.com/56751947/192301482-ddfa122f-3bd6-42a4-b3e4-f55c563582b8.jpg)

### CLASSIFICATION SCORES

![Screenshot 2022-09-26 072425](https://user-images.githubusercontent.com/56751947/192301732-24461943-4bea-401c-9af3-be036c79a873.jpg)



