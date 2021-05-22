# parkinson-python-ml
Parkinson's Disease Detection Using Machine  Learning With Python
What is Parkison Disease (PD) :
Parkinson’s Disease(PD) is a nervous system disorder that affects movement. It mainly affects 
the motor system. As the disease worsens, non-motor symptoms become more common. And 
the worst thing is there is no cure for Parkinson’s disease.
So the best possible way is to get the disease detected in its early stage and start the 
treatment.And here comes the use of Machine Learning Algorithms. ML algorithms can 
determine the health status of the individual.
Data Source :
I have used data set from UCI Machine Learning Repository. Follow the link to download 
the data set: https://archive.ics.uci.edu/ml/datasets/parkinsons
The data set contains 22 features based on which we will classify the health status.It is set 0 for 
healthy and 1 for PD. The data is in ASCII CSV format.
Overview :
At first we will import the essential libraries for our model i.e numpy, matplotlib, pandas .Then 
we import the data set through the python library pandas. And divide the data set into 
Dependent (y) and Independent (X) variable.
Now for training and testing the model we first split the data into training set and test set. For 
splitting the data set into train and test we use model selection library from Scikit-learn .We set 
the test size at 0.2 so that 20% data goes to test set and 80% of the data goes to training.
As we need exact predictions from the model, we need to feature scale the data.Therefore, to 
feature scale our data we use the Standard Scaler class of preprocessing library from Scikit-learn.Then here comes our most important part of our model which will help us to obtain optimum 
results from our model i.e we apply Principal Component Analysis(PCA).
Principal Component Analysis :
The main idea of principal component analysis (PCA) is to reduce the dimensionality of a data set 
consisting of many variables correlated with each other, either heavily or lightly, while retaining 
the variation present in the data set, up to the maximum extent. PCA extracts new independent 
variables from our data set that explain the most of the variance of the data set, i.e regardless of 
the dependent variable. And that makes PCA an unsupervised model.
And then finally train our data in Machine Learning Models and obtain the results.
