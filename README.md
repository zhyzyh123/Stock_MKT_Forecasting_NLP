## Chinese Stock Market Forecasting based on NLP Sentiment Analysis
This project is a sentiment analysis of the Chinese Stock investors' online comments from ___ to ___. It is built using `NLTK`, `spaCy`, `scattertext` and `markovify` libraries. This project was done for the course "Introduction to NLP in Python" at Brandeis University.

For motivations, technical details, etc please see the project `report.pdf` file. 
- For information about the website for getting comments: http://guba.eastmoney.com/list,zssh000001.html.


These project contaisn the following steps:
> 1. Use `urllib` package for accessing the website via `python` and then store the comments in `Excel`; choose xxxx of them to do the further sentiment polarity tagging by our own, and we tagged the `bullish comment` as `positive` and tagged the `bearish comment` as `negative`, the remaining comments(which cannot see the clear sentiment polarity) were tagged as `neutral`；then we use numbers to represent the sentiment polarity -- `1` for `pisotive`, `-1` for `negative` and `0` for `neutal`.
> 2. Split the data by 9:1 (90% data for training and 10% for testing), extract features, go through data visualization and then build up own claasifiers by using `LinearSVC`,`BernouliNB`,`NuSVC`,`LogisticRegression`,`MultinomialNB`,`DecisionTree` and `RandomForest`. 
> 3. Implement comparisons of `Accuracy`, `Recall` and `Precision`...among thses machine learning classifiers we mentioned before, choose the best one as our final model.
> 4. Apply our final model to test the whole data set. In this case, for each day, we will get a number of positive comments for that day's stock market (M1) and a number of negative comments for that day's stock market (M2). And we are able to use these two numbers to get an Investor Sentiment Index (we will explain how we define it in our project report).
> 5. Finally, we will use our Investor Sentiment Index to compare with the .


This project contains the following files: 
* The `speeches` folder contains the files with the Christmas speeches from 1975 to 2018 in `txt` format.
* The `Speech` class creates the object Speech with the information for a given speech.
* The `Corpus` class creates the corpus object that contains the speeches to be analyzed. This class uses the files inside the `speeches` folder. This file also contain the methods to perform the lexical analysis of the created corpus. An example of output of this lexical analysis can be found inside the output.txt file.
* The `visualize.py` script creates interactive HTML visualization from TF-IDF measures using `scattertext` library. The visualization files are stored inside the visualization folder.
* The `markov.py` script generates random speeches automatically with markov models.
* The `crawler.py` script is the script that was used to extract the speeches from http://www.casareal.es. The URLs of the speeches are in this file. Please, note that this process took several tries and was humanly supervised (due to timeout errors from the server and to label each speech with its corresponding year).


The files that have a main method than can be executed are: 
* `corpus.py` (creates several corpus objects for different time periods of time and calls the radiography method in order to get their lexical analysis)
* `visualize.py` (creates an instance of the corpus class and generates several visualization files with TF-IDF measures using the scattertext library)
* `markov.py` (creates an instance of the corpus class and generates randomly generated speeches following the markov chain).




This project requires the following libraries:
* `nltk`
* `spacy`
* `pandas`
* `scattertext`
* `markovify`
* `os`
* `re`
* `time`
* `newspaper`
* `matplotlib`
* `TuShare`
