import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

distorsions = []
inertias = []

class Kmeans:
    
    def __init__(self, nb_test, k, X, x_column, y_column) -> None:
        self.list_of_TP = []
        self.list_of_FP = []
        self.list_of_centroid_params = []
        self.nb_test = nb_test
        self.X = X
        self.k = k
        self.centroid = self.X.sample(n = self.k)
        self.x_column = x_column
        self.y_column = y_column
        self.centro_optimisation = None
        self.initialize_centroid()
    
    def initialize_centroid(self):
        n_dims = self.X.shape[1]
        centroid_min = self.X.min().min()
        centroid_max = self.X.max().max()
        centroids = []

        for centroid in range(self.k):
            centroid = np.random.uniform(centroid_min, centroid_max, n_dims)
            centroids.append(centroid)

        centroids = pd.DataFrame(centroids, columns = self.X.columns)
        return centroids
    
    def model(self, df):
        self.centroid = self.X.sample(n = self.k) 
        for _ in tqdm(range(self.nb_test)): 
            dict = {} 
            for i in self.centroid.index:
                dict[i] = []
            
            for i, row_c in self.X.iterrows():
            
                for index, row in self.centroid.iterrows():  
                    soustrac_0 = (self.centroid.loc[index, self.x_column] - row_c[self.x_column])**2
                    soustrac_1 = (self.centroid.loc[index, self.y_column] - row_c[self.y_column])**2
                    d = np.sqrt(soustrac_0+soustrac_1)
                    dict[index].append(d)


            self.df_a = pd.DataFrame(dict) 
            self.df_a['minvalue'] = self.df_a.idxmin(axis=1)

            df['value'] = self.df_a['minvalue']
            
            self.centroid = self.optimisation(df)
            
            self.plot_clusters(df)
            
    def optimisation(self, df):
        for i in range(len(df.value.unique())):

                cluster_1_x = df[df['value'] == df.value.unique()[i]][self.x_column].mean().astype(int)
                cluster_1_y = df[df['value'] == df.value.unique()[i]][self.y_column].mean().astype(int)
                
                self.centroid.loc[self.centroid.index[i], self.x_column] = cluster_1_x
                self.centroid.loc[self.centroid.index[i], self.y_column] = cluster_1_y
        
        return self.centroid
    
    @staticmethod
    def color_maker(i, df, colors):
        for iterable in df['value'].unique():
            if df.loc[i,'value'] == iterable:
                color = colors[np.where(df['value'].unique() == iterable)[0][0]]
        return color
    
    def plot_clusters(self, df):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        
        for i in range(len(self.X)):
            color = Kmeans.color_maker(i, df, colors[: self.k])
            plt.scatter(df.loc[i, self.x_column], df.loc[i, self.y_column], c = color)
        plt.scatter(self.centroid[self.x_column], self.centroid[self.y_column], c = 'black')
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.show()
        
