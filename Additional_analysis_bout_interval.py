import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score

def cluster_data(values,number_clusters):
    data = np.array(values).reshape(-1, 1)
    k = number_clusters
    km = KMeans(n_clusters=k, random_state=0).fit(data)
    labels = km.labels_
    centers = km.cluster_centers_.flatten()
    print(f'Cluster centers: {centers}')
    db = davies_bouldin_score(data, labels)
    print(db)
    


# # 1) Elbow method
# inertias = []
# Ks = range(1, 10)
# for k in Ks:
#     km = KMeans(n_clusters=k, random_state=0).fit(data)
#     inertias.append(km.inertia_)

# plt.figure()
# plt.plot(Ks, inertias, marker='o')
# plt.xlabel('k')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.show()

# # 2) Fit KMeans with chosen k (e.g. 3)
# k = 3
# km = KMeans(n_clusters=k, random_state=0).fit(data)
# labels = km.labels_
# centers = km.cluster_centers_.flatten()

# # 3) Build a DataFrame and print
# df = pd.DataFrame({'value': values, 'cluster': labels})
# print(df)

# # 4) Optional: scatter-plot the clusters along a line
# plt.figure()
# # jitter y for visibility
# y = np.zeros_like(values) + np.random.uniform(-0.02, 0.02, size=len(values))
# plt.scatter(values, y, c=labels, cmap='tab10', s=50)
# plt.scatter(centers, [0]*len(centers), marker='X', s=200, edgecolor='k')
# plt.yticks([])
# plt.xlabel('Value')
# plt.title(f'KMeans clusters (k={k})')
# plt.show()

# print(f'Cluster centers: {centers}')


if __name__ == "__main__":
    initial_file = r"U:\Users\Ruthi\2025\head drumming_Silvia.xlsx"
    sheet_name = "BMR22"
    values = pd.read_excel(initial_file,sheet_name= sheet_name)
    time_interval = values['inter_bouts(sec)'].dropna()
    number_clusters = 2
    cluster_data(time_interval,number_clusters)