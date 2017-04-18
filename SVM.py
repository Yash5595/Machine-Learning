import numpy as np
from __future__ import print_function
from sklearn.svm import libsvm
from sklearn.datasets import load_svmlight_file

#loading the test data
X,Y = load_svmlight_file(f='train_raw', n_features=16, multilabel=False, zero_based='auto', query_id=False, dtype=np.float64)
X = X.toarray()

#To train the svm.
[support, sv, nsv, coeff, intercept, proba, probb, fit_status] = libsvm.fit(X, Y, svm_type=0, kernel='rbf',
        gamma=0.0001, coef0=0, tol=1e-8, C=7.5, nu=0.5, epsilon=0.1, max_iter=-1, random_seed=0)

#Dumping model in a pickle file
def save_model(file_name, model_list):
    import pickle
    with open(file_name, 'wb') as fid:
         pickle.dump(model_list, fid)


#Loading the model from the pickle file           
def load_model(file_name):
    import pickle
    with open(file_name, 'rb') as fid:
         model = pickle.load(fid)
    return model

#Stacking the model parameters
model = [support, sv, nsv, coeff, intercept, proba, probb, fit_status]      
            
#Saving the model in the file model.pkl            
save_model('model.pkl', model)

#Saved model is loaded back during the testing.
[support, sv, nsv, coeff, intercept, proba, probb, fit_status] = load_model('model.pkl')

inputs= np.genfromtxt('test_set.csv',delimiter=' ')

#To make predictions from the trained model
dec_values = libsvm.predict(inputs, support, sv, nsv, coeff, intercept, svm_type=0,kernel='rbf',gamma=0.0001,coef0=0)
print (nsv)
for i in dec_values:
    print (i)
