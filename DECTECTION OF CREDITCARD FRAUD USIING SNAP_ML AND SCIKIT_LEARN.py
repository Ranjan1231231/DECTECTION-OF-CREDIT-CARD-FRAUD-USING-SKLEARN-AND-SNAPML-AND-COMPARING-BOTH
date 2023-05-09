import  matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.utils.class_weight import  compute_sample_weight
from sklearn.metrics import roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier


###

#IMPORTING,READING THE DATA,PREPROCESSING THE DATA

###
#IMPORTING THE DATA FROM CSV FILE
raw_data=pd.read_csv('creditcard.csv')
# print("There are "+str(len(raw_data))+" observations in the credit card fraud dataset.")
# print("There are "+str(len(raw_data.columns))+" Variables in the datasets.")

#READING THE DATA
# print(raw_data.head())


#In practice, a financial institution may have access to a much larger dataset of transactions. To simulate such a case, we will inflate the original one 10 times.


n_replicas=10
big_raw_data=pd.DataFrame(np.repeat(raw_data.values,n_replicas,axis=0),columns=raw_data.columns)
# print("There are " + str(len(big_raw_data)) + " observations in the inflated credit card fraud dataset.")
# print("There are " + str(len(big_raw_data.columns)) + " variables in the dataset.")
# print(big_raw_data.head())

#For confidentiality reasons, the original names of most features are anonymized V1, V2 .. V28. The values of these features are the result of a PCA transformation and are numerical. The feature 'Class' is the target variable and it takes two values: 1 in case of fraud and 0 otherwise.

#GET THE SET OF DISTINCTIVE CLASS
labels=big_raw_data.Class.unique()
size=big_raw_data.Class.value_counts().values
fig,ax=plt.subplots()
# ax.pie(size,labels=labels,autopct='%1.3f%%')
# ax.set_title('Target Variable Value Counts')
# plt.show()



# plt.hist(big_raw_data.Amount.values,6,histtype='bar',facecolor='g')
# plt.show()


# print("Minimum amount value is ",np.min(big_raw_data.Amount.values))
# print("Maximum amount values is ",np.max(big_raw_data.Amount.values))
# print("90% of the transaction have an amounnt less or equal than ", np.percentile(raw_data.Amount.values,90))

#DATASET PREPROCESSING
# data preprocessing such as scaling/normalization is typically useful for
# linear models to accelerate the training convergence
# standardize features by removing the mean and scaling to unit variance
big_raw_data.iloc[:,1:30]=StandardScaler().fit_transform(big_raw_data.iloc[:,1:30])
data_matrix=big_raw_data.values


# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X=data_matrix[:,1:30]

# y: labels vector
y = data_matrix[:, 30]

#data normalisation
X=normalize(X,norm='l1')
# print('X.shape=',X.shape,'y.shape=',y.shape)

#Dataset train/test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
# print('X_train.shape= ', X_train.shape, ' Y_train.shape=',y_train.shape)
# print('X_test.shape=',X_test.shape,"Y_test.shape",y_test.shape)

#BUILDING A DECISISON TREE CLASSIFIER MODEL
# compute the sample weights to be used as input to the train routine so that
# it takes into account the class imbalance present in this dataset

w_train=compute_sample_weight('balanced',y_train)
sklearn_dt=DecisionTreeClassifier(max_depth=4,random_state=35)


###


# Train a Decision Tree Classifier using scikit-learn



###
# t0=time.time()
# sklearn_dt.fit(X_train,y_train,sample_weight=w_train)
# sklearn_time=time.time()-t0
# print("[Scikit-Learn] Training time(s):{0:.5f}".format((sklearn_time)))

###



#BUILD A DECISION TREE CLASSIFIER MODEL WITH SNAP ML



###
# from snapml import DecisionTreeClassifier
# snapml_dt=DecisionTreeClassifier(max_depth=1,random_state=45,n_jobs=4)
# # train a Decision Tree Classifier model using Snap ML
# t0=time.time()
# snapml_dt.fit(X_train,y_train,sample_weight=w_train)
# snapml_time=time.time()-t0
# print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))


#Evaluate the Scikit-Learn and Snap ML Decision Tree Classifier Models
# training_speedup=sklearn_time/snapml_time
# print('[Decision Tree Classifier] Snap ML vs. Scikit-Learn speedup : {0:.2f}x '.format(training_speedup))
# #sklearn prediction
# sklearn_pred=sklearn_dt.predict_proba(X_test)[:,1]
# ## evaluate the Compute Area Under the Receiver Operating Characteristic
# # Curve (ROC-AUC) score from the predictions
# sklearn_roc_auc=roc_auc_score(y_test,sklearn_pred)
# print('[Scikit-Learn] ROC-AUC score : {0:.3f}'.format(sklearn_roc_auc))

#snapml_prediction
# snapml_pred=snapml_dt.predict_proba(X_test)[:,1]
# snapml_roc_auc=roc_auc_score(y_test,snapml_pred)
# print('[Snap ML] ROC-AUC score : {0:.3f}'.format(snapml_roc_auc))


#BUILDING A SUPPORT VECTOR MACHINE MODEL WITH SCIKIT-LEARN
from sklearn.svm import LinearSVC
sklearn_svm=LinearSVC(class_weight='balanced',random_state=31,loss="hinge",fit_intercept=False)
# train a linear Support Vector Machine model using Scikit-Learn

t0=time.time()
sklearn_svm.fit(X_train,y_train)
sklearn_time=time.time()-t0
print("[Scikit-Learn] Training time (s):  {0:.2f}".format(sklearn_time))

#Build a Support Vector Machine model with Snap ML

from snapml import SupportVectorMachine

# in contrast to scikit-learn's LinearSVC, Snap ML offers multi-threaded CPU/GPU training of SVMs
# to use the GPU, set the use_gpu parameter to True
#snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, use_gpu=True, fit_intercept=False)


# to set the number of threads used at training time, one needs to set the n_jobs parameter
snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)

# train an SVM model using Snap ML

t0=time.time()
model=snapml_svm.fit(X_train,y_train)
snapml_time=time.time()-t0
print("[Snap ML] Training time (s):  {0:.2f}".format(snapml_time))

#Evaluate the Scikit-Learn and Snap ML Support Vector Machine Models
training_speedup=sklearn_time/snapml_time
print('[Support Vector Machine] Snap ML vs. Scikit-Learn training speedup : {0:.2f}x '.format(training_speedup))
sklearn_pred=sklearn_svm.decision_function(X_test)
# evaluate accuracy on test set
acc_sklearn=roc_auc_score(y_test,sklearn_pred)
print("[Scikit-Learn] ROC-AUC score:   {0:.3f}".format(acc_sklearn))

snapml_pred=snapml_svm.decision_function(X_test)

# evaluate accuracy on test set
acc_snapml=roc_auc_score(y_test,snapml_pred)
print("[Snap ML] ROC-AUC score:   {0:.3f}".format(acc_snapml))


#evaluating the quality of the SVM models trained above using the hinge loss metric
# get the confidence scores for the test samples
sklearn_pred=sklearn_svm.decision_function(X_test)
snapml_pred=snapml_svm.decision_function(X_test)

# import the hinge_loss metric from scikit-learn

from sklearn.metrics import hinge_loss

#evaluate the hinge loss from the predictions

loss_snapml=hinge_loss(y_test,snapml_pred)
print("[Snap ML] Hinge loss:   {0:.3f}".format(loss_snapml))


#evaluate thte hinge loss metric from the predictions
loss_sklearn=hinge_loss(y_test,sklearn_pred)
print("[Scikit-learn] Hinge loss: {0:3f}".format(loss_sklearn))


