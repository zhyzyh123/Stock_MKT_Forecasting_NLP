## Chinese Stock Market Forecasting based on NLP Sentiment Analysis

#### Group Members: Yuehuan Zhang, Ilsa Wu, Ran Dou, Xuanbo Mao


#### Introduction:

This project is a sentiment analysis of the Chinese Stock investors' online comments from `2018-08-06` to `2019-12-17`. It is built using `Scikit-learn`, `jieba`, `urllib2` and etc. This project was done for the course "Introduction to NLP in Python" at Brandeis University.


- For motivations, technical details, etc. please see the project `report.pdf` file. 
- For information about the website for getting comments: http://guba.eastmoney.com/list,zssh000001.html. This website is an online talking bar for Chinese stocks which had IPO in Shanghai Stock Exchange. 
- For information about the `SSE Composite Index`, please see the website: https://en.wikipedia.org/wiki/SSE_Composite_Index.
- For information about the library `jieba`, it is like the `nltk` library, which is especially for Chinese Natural Language Processing. You should download this by command `pip3 install jieba`. 
>  "Jieba" (Chinese for "to stutter") Chinese text segmentation: built to be the best Python Chinese word segmentation module.

>  More information could be found at https://github.com/fxsjy/jieba.
- For Information about `tushare` library, information could be found at https://github.com/waditu/tushare. This libarary could do data collection, data cleaning and data storage for Chinese stock market's financial data. 



#### These project contains the following steps:

* 1. Use `urllib` package for accessing the website via `python` and then store the comments in `Excel`; choose thousands of them to do the further sentiment polarity tagging by our own, and we tagged the `bullish comment` as `positive` and tagged the `bearish comment` as `negative`；then we use numbers to represent the sentiment polarity -- `1` for `positive`, `-1` for `negative`.  

* 2. Go through the data by cross-validation with `scikit-learn` - `KFold(n_splits=5, shuffle=True, random_state=42)`, go through data vectorization (`count`,`binary`,`Tfidf`) and then build up own classifiers by using `LinearSVC`, `BernouliNB`, `NuSVC`, `LogisticRegression`, `MultinomialNB`, `DecisionTree` and `RandomForest`. 

* 3. Implement comparisons of `Accuracy`, `Positive Recall`, `Negative Recall`, `Positive Precision` and `Negative Precision` among these machine learning classifiers we mentioned before, choose the best one as our final model. 

* 4. Apply our final model to test the whole data set. In this case, for each day, we will get a number of positive comments for that day's stock market (M1) and a number of negative comments for that day's stock market (M2). And we are able to use these two numbers to get an Investor Sentiment Index - `BI = ln [(1 + M1)/(1+M2)]` (we will explain how we define it in our project report).

* 5. We will conduct a data visualization for Investor Sentiment Index and compare this with the whole SSE Composite index. See whether there is a similar trend between Investor Sentiment Index and SSE Composite index. 

* 6. Finally, we will add the Investor Sentiment Index as an exogenous variable for the time series forecasting of SSE Composite index. 




#### This project contains the following files: 

* The `data` folder contains the files we need to process: `annotation_data.xlxs` - tagged comments; `annotation_data_after_segment` folder - contains `negative.txt` and `positive.txt` which are tagged comments after segment and are in `txt` format; `raw data` folder include the whole data set which were downloaded from website by using `urllib`.
* The `crawler.py` script accessed the target website http://guba.eastmoney.com/list,zssh000001.html, used to extract the comments from this website and stored the raw data.
* The `data_processing.py` script went through the `annotation_data.xlxs` and created the tagged corpus objects(`negative.txt` and `positive.txt`) that contains the comments to be trained and tested. 
* The `classify.py` generated different classifiers, compared the accuracy, precision and recall of these classifiers, and built up the Investment Sentiment Index based on the best model we chose. The results of Investment Sentiment Index are stored in `date_BI.csv`. In this file, we also do some data visualization for the comparision between Investor Sentiment Index and SSE Composite index. These results are stored inside the `Visualization` folder. 
* The `model_comparison.py` script generated the results of comparison between different models.







#### This project requires the following libraries:
* `urllib2`
* `xlwt`
* `jieba`
* `collections`
* `TuShare`
* `sklearn`
* `data_processing`
* `classify`
* `os`
* `xlrd`
* `math`
* `pandas`
* `matplotlib`
* `scipy`
* `statsmodels`
* `numpy`
* `time`
* `datetime`
* `sys`

