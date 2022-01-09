# knearest

https://www.youtube.com/watch?v=s-9Qqpv2hTY


K- nearest neighbour classification is a supervised machine learning method that you can use to classify instances based on the arithmetic difference between features in a label data set.  In the coding demonstration for this technique you are going to see how to predict whether a core has an automatic or manual transmission based on its number of gears and carburetors. K nearest neighbor works by memorizing observations within a label test set to predict classification labels for new incoming unlabeled observations. The algorithm makes predictions based on how similar training observations are to the new incoming observations.
The more similar the observations value the more likely they will be classified with the same label. Popular use cases for the K nearest neighbour algorithm are stock price prediction, recommendation systems, predictive trip planning and credit risk analysis.

The K nearest neighbor model has a few assumptions

The K nearest neighbor model has a few assumptions those are that the 

Data set has little noise 

Data is labeled that it contains only relevant features and the  

Data set has distinguishable subgroups you want to be sure to avoid using K nearest neighbour algorithm on large data sets cause it will probably take away too long 


let’s use Python to apply the K nearest neighbor algorithm you will start by importing our libraries so we are going to need numpy pandas inside by for this demonstration or side B 
we will also import net plot clip for our data visualisation and to read or data in we are going to use URL name so we will say import URL delay and then for the modeling itself we will use scikit learn so we we will say import SK learn and from SK learns neighbor module we want to import K neighbor’s classifier so we will say import SK learn dot neighbors and then we want to import K neighbor classifier let’s be sure to import our neighbors module itself so we will say 
from a scale and import neighbors for our pre-processing of our data we want to import the pre-processing module so from SK learn import preprocessing and we also want to import the train test split function I am going to show you how to use this to split our data into test and training sets so we say from a SK learn on import our cross-validation debts the module
 that has the tool and we want to import train slash test split and to evaluate our module we will import a scikit learn matrix so from 

import numpy as np
import pandas as pd 
	
	Import scipy 
	Import matplotlib.pyplot as plt
	From pylab import reParams
	Import urllib
	Import sklearn 
	From sklearn.neighbors import KNeighborsClassifier 
	From sklearn import neighbors
From sklearn import preprocessing
From sklearn.cross_validation import train_test_split

From sklearn import metrics

SK learn import matrix run that on the fifth line up from the bottom it should be from SK learn neighbors and poor K neighbors classifier and when we run that we have got all our libraries 


Splitting your data into test and training datasets 
now let’s set our plotting parameters for the jupiter notebook like I said you are going to use our NT course data set so we will load it like we hoping throughout the previous video lectures and then to use K nearest neighbor you should have a label data set needle we are going to use the end variable as our target this variable label is a car’s either having an automatic transmission or a manual transmission.  for this analysis you are going to use the variables mpg displacement HP and weight as predictive features in our model. we are going to build a model that predicts a car’s transmission type based on values in these four fields. I pick these variables because they each hold information that is relevant to whether a car has an automatic or a manual transmission and because they have each have distinguishable subgroups so we will call our subset X Prime and you will set that equal to course dot I lock and we will use our special indexer to select and retrieve our columns.  in this case those are columns 1 3 4 & 6 and then we will say dot values because we want to access the values in those columns we also need to set our target variable we will call that Y and we will say Y is equal to car dark IX?  and the n variable is the column with the index number 9 so we will say 9 and then

address = ‘C: / Datascience files/Ex_Files_Python_Data_Science_EssT/Exercise Files/Ch01/01_05/mtcars.csv’

Cars = pd.read_csv(address)

Cars.columns = [ ‘car_names’, ‘mpg’, ‘cyl’, ‘disp’, ‘hp’, ‘drat’, ‘wt’,’qsec’,’vs’, ‘am’, ‘gear’, ‘carb’]

X_prime = cars.ix [ :,(1,3,4,5) ]. Values

Y = cars.ix [:,9].values

Y = cars.ix [:, 9].values

C:\ProgramData\Anaconda2\lib\site-packages\ipykernel_launcher.py:7: DeprecationWarning:
.ix is deprecated. Please use 
.loc for label based indexing or 
.iloc for positional indexing

See the documentation here:
http://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate_ix
      Import sys

dot values let me run 



that before we can implement the K nearest neighbor algorithm we need to scale our variables so we will create a scale data set and we will call it X and then we will use psych it learns pre processing tools if we use the scale function so pre processing dot scale and then we will pass in our X prime object that scale is our variables now I am going to split the data into the test and training set you use the training set for training the model and the test set for evaluating the models performance to do this you will use Scikit learn selection tools then you will use the train test slip function the train test with slip function breaks the original data set into a list of train test splits so we will say train test split we will pass in our X data in our Y data and for the model outputs those are going to be explained X test Y trained in Y test we also need to specify some model parameters so we will say test size equal to 0.33 this tells the function that we want to split our data so that 33% of it goes into the test set and 77% of it goes into the training set and let’s also pass in the parameter random state equal to 17 since the function splits the data randomly we need to set that speed by passing this argument in and that will allow you to reproduce the same results as you see here on my computer we run that and 

In [ 7 ]: X = preprocessing.scale(X_prime)

In [ 8 ]: X_train, X_test, y_train, y_test= train_test_split (X, y, test_size = .33, random_state = 17)



Building and Training Your Model With Training Data

now let’s build our model the first thing we need to do is instantiate K nearest neighbor object we will call a CLF and we’ll set that equal to neighbor’s dot K neighbors classifier next we call the fit method off of the model and pass in X_ train is our training data and Y train is our target variable we say CLF not fit and then explain Y_train then let’s just print it out and now what we see here is our model parameters all printed out 

In [ 9 ]: clf = neighbors.KNeighborsClassifier()
	Clf.fit (X_train, y_train)
	print(clf)
KNeighborsClassifier(algorithm=’auto’, leaf_size=30, metric=’minkowski’,)
	         metric_params=None, n_jobs=1, n_neighbors=5, p=2,
	         

Evaluating your model’s predictions against the test dataset

now let’s evaluate the models predictions against the test data set just to make this easier to explain I am going to rename our Y set to Y expect presenting our expected level values so Y expect equals to Y test then I am going to create another variable called Y print this variable is going to contain the labels that our models predicts for the Y variable so we will say Y press and then we will write the name of our model then we will call the predict method off of it and pass in our test data set so X tests to score the model I will use psychic it learns classification report function that is part of the matrix module so we will say matrix dot classification report and we will pass in Y expect and Y pred and then 

In [ 10 ]: y_expect = y_test
	  Y_pred = clf.predict (X_test) 
	  
 	  Print { metrics.classification_report (y_expect, y_pred) }

let’s just print this whole thing out so we will call the print function on the whole thing and there we have some model results now I am going to take you into the other skin to show you what those mean 

K-Nearest Neighbor Example

as you remember from the K means demonstration 
recall is a measure of your model’s completeness what these results are saying is that 

of all the points that were labeled 1 only 0.67 of those results were returned were truly relevant

 and of the entire dataset 82 percent of the results that were returned or truly relevant high position and low recall generally means that there are fewer results returned but many of the labels that are predicted are returned correctly in other words high accuracy but low completion that is it for instance based learning hold on because 
Next I am going to show you how to use python for network analysis       

High Precision + low recall = Few results returned, but many of the label predictions returned were correct. because they have each have
