import csv
import numpy as np

i = 0
inputs = []
ni, nh, no = 17, 30, 26
out = []
length = 0

inputs = np.genfromtxt('train_data.csv', delimiter=',')
inputs = np.concatenate((inputs, np.ones((len(inputs), 1))), axis = 1)

#training data
length = 2*len(inputs)/3
inp_train = np.array(inputs[:length])

out = np.zeros((len(inputs), 26))
iter1 = 0

outputs = np.genfromtxt('train_labels.csv', delimiter=',')


for i in outputs:
    out[iter1][i-1]+=1
    iter1+=1

out_train = out[:2*len(out)/3]

#randomly generating weights
np.random.seed(1)
w1 = np.random.randn(17, 30)
w2 = np.random.randn(30, 26)


def softmax(x):
    x = x - np.max(x, axis = 1, keepdims=True)
    expo = np.exp(x)
    hx = expo/np.sum(expo, axis=1, keepdims=True, )
    return hx

def sigmoid(z):
    #sigmoid function
    return 1/(1+np.exp(-z))

def sigmoidprime(z):
    return sigmoid(z)*(1-sigmoid(z))

def train():
    iter2 = 0
    prediff = []
    for i in range(0, inp_train):
        
        a = i.reshape(1, 17)
        b = out_train[iter1].reshape(1, 17)
        global w1
        global w2
        lamda = 0.001
        learn = 0.005
        iter2 += 1
        
        #200 iterations
        for i in range (0, 200):
            l0 = inp_train
            l1 = sigmoid(np.dot(l0,w1))
            l2 = softmax(np.dot(l1,w2))
            
            #calculating error in layer 2
            l2_error = y1 - softmax(np.dot(x2, w2)
            l2_delta = l2_error*(sigmoidprime(l2)) + (lamda*W2)
            
            #calculating error in layer 1                        
            l1_error = l2_delta.dot(syn1.T)
            l1_delta = l1_error * sigmoidprime(l1) + (lamda*W1)
            
            #updating weights (backpropagation)                   
            W2 += l1.T.dot(l2_delta)*learn
            W1 += l0.T.dot(l1_delta)*learn     
    print prediff
train()

                                    
def test(x):
    x1 = np.dot(x, w1)
    x2 = reluvec(x1)
    a = softmax(np.dot(x2, w2))
    for a in np.argmax(a, axis = 1):
        print a+1
test(inputs)


inp_test = np.genfromtxt('test_data.csv', delimiter=',')
inp_test = np.concatenate((inp_test, np.ones((len(inp_test), 1))), axis = 1)
inp_test.shape
test(inp_test)




