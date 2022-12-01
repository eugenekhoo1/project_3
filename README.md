# Project 3: Using NLP to classify user posts

## Problem Statement

## Methodology
In order to build the classifier we scraped the r/Anxiety and r/Depression subreddit forums to train our models. A total of 6,000 posts were pulled (3,000 each) with 2,754 (anx) and 2,351 (dep) remaining after removing empty posts. 

We then processed the data to prepare it for NLP. 
1. Removed punctuation
2. Removed word strings with numbers
3. Removed stopwords (using nltk)
4. Lemmatized tokens

We applied two methods of vectorization while limiting the maximum features to 10,000 for both:
1. CountVectorizer (ngram_range=(1,2))
2. TFIDFVectorizer

We ran these two sets of data through three models:
1. Logistic Rgression
2. Multinomial Naive Bayes
3. Bernoulli Naive Bayes

## Results
From the six models, we notice that TF-IDF vectorized dataset perform better than count vectorized ones, while logistic regressoins outperform naive bayes models.
The ROC curve shows that this is the case on average across all threshold levels.
 insert roc curve

 The best performing model is the logistic regression model on TF-IDF vectorized data, with train scores of 93.7% and test scores of 86.3%

insert classification matrix

The top 10 features in predicting classes are also shown below.
insert graph

## Production
With our results, we deployed the model onto streamlit to classify users posts into Anxiety and Depression.
insert link

## Conclusion
While the model is able to function in production, accuracy scores may be improved by using a larger data set. This can be accomplished by accessing other forums as long as their data is labelled.