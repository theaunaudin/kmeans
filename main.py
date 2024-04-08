from pandas import read_csv
import kmeans
df = read_csv('./Country-data.csv')

k = 5
columns = ['country', 'income']
X = df[columns]


km = kmeans.Kmeans(nb_test = 100, k = k, X = X, x_column = columns[0], y_column = columns[1])
km.model(df)
km.plot_clusters(df)