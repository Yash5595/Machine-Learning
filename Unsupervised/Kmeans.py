import numpy as np
from __future__ import print_function 
inputs = np.genfromtxt('train_data.csv', delimiter=' ')

cent = []
def Kmeans(data,k):
    centroids = []
    iters = 0
    n = len(data)
    clusters=[[] for i in range(k)]
    output = np.zeros(n)
    iters = 0
    
    #initializing random centroids 
    centroids = rand_centroids(data, centroids, k)
    global cent 
    cent = centroids
    
    while(iters<100): 
        iters +=1
        clusters,outputs = euclidean(data,centroids,clusters,output)
        index =0
        
        #updating the current centroid to the mean of the points in the cluster
        for cluster in clusters:
            centroids[index] = np.mean(cluster,axis=0)
            index +=1
    #saving the output labels in a csv file 
    np.savetxt("Kmlabel.csv", output, delimiter=",")
    return output
    
#function to calculate the euclidean distance.
def euclidean(data,centroids,clusters,output):
    n = len(data)
    for i in range(len(data)):
        index =-1
        x=-1
        minimum=10000
        
        #calculates the euclidean distance between the input point and means of the cluster and then saving 
        #the minimum of those distances.  
        for k in range(len(centroids)):       
            x = np.linalg.norm(np.subtract(data[i],centroids[k]))**2   
            if(x<minimum):
                minimum =x
                index=k
                
        # Now this point belongs to the corresponding cluster and the index of this 
        #cluster is stored in output array.       
        output[i]=index
        clusters[index].append(data[i])
        
    #if any cluster remains empty after an iteration then a random point from data set is assigned 
    #to it so that a mean exists for this cluster
    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, n, size=1)])
    return (clusters,output)
    
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
m=[cent]
 
#Saving the model in the file model.pkl
save_model('modelK.pkl', m)
 
#These saved model can be loaded back during the testing.
[cent] = load_model('modelK.pkl')
 
#Now use these values to initialize 
n = Kmeans(inputs,k)