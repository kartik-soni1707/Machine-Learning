# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing dataset
dataset=pd.read_csv("Credit_Card_Applications.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)
#Train Som
from minisom import MiniSom
Som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
Som.random_weights_init(X)
Som.train_random(X, num_iteration=100)
#Visualizing the results
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(Som.distance_map().T)
colorbar()
markers=['o','s']
colors=['red','green']
for i,x in enumerate(X):
    w=Som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[Y[i]],
         markeredgecolor=colors[Y[i]],
         markerfacecolor="None",
         markersize=10,
         markeredgewidth=2)
show()
#Finding out liers who got away
mappings=Som.win_map(X)
frauds=np.concatenate((mappings[(7,7)],mappings[(6,6)]),axis=0)
frauds=sc.inverse_transform(frauds)
