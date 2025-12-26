# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# load the data
df = pd.read_csv('airlines.csv')

# Rename columns
# df = df.rename(columns={'Statistics.# of Delays.Carrier': 'Delays.Carrier', 'Statistics.# of Delays.Late Aircraft': 'Delays.Late', 'Statistics.# of Delays.National Aviation System': 'Delays.NAS', 'Statistics.# of Delays.Security': 'Delays:Security', 'Statistics.# of Delays.Weather': 'Delays.Weather', 'Statistics.Carriers.Names': 'Carriers.Names', 'Statistics.Carriers.Total': 'Carriers.Total', 'Statistics.Flights.Cancelled': 'Flights.Cancelled', 'Statistics.Flights.Delayed': 'Flights.Delayed', 'Statistics.Flights.Diverted': 'Flights.Diverted', 'Statistics.Flights.On Time': 'Flights.On_Time', 'Statistics.Flights.Total': 'Flights.Total', 'Statistics.Minutes Delayed.Carrier': 'Min_Delay.Carrier', 'Statistics.Minutes Delayed.Late Aircraft': 'Min_Delay.Late', 'Statistics.Minutes Delayed.National Aviation System': 'Min_Delay.NAS', 'Statistics.Minutes Delayed.Security': 'Min_Delay.Security', 'Statistics.Minutes Delayed.Total': 'Min_Delay.Total', 'Statistics.Minutes Delayed.Weather': 'Min_Delay.Weather'})

# Drop unnessary columns
df.drop(['Time.Label', 'Time.Month Name'], axis=1, inplace=True)

print("=== First 5 Rows ===")
print(df.head(), "\n")
print("=== Description Statistics ===")
print(df.describe(include='all'), "\n")
print("=== Dataframe Info ===")
df.info()

# Correlation Matrix Heatmap
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
plt.figure(figsize=(12, 10))
plt.imshow(corr, interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# Feature engineering
df_feat = df.copy()
# Delay rates per flight
delay_cols = {
    'carrier_delay_rate': 'Statistics.# of Delays.Carrier',
    'weather_delay_rate': 'Statistics.# of Delays.Weather',
    'nas_delay_rate': 'Statistics.# of Delays.National Aviation System',
    'security_delay_rate': 'Statistics.# of Delays.Security',
    'late_aircraft_delay_rate': 'Statistics.# of Delays.Late Aircraft'
}
for feat, col in delay_cols.items():
    df_feat[feat] = df_feat[col] / df_feat['Statistics.Flights.Total']
# Average delay minutes per flight
df_feat['avg_delay_minutes'] = df_feat['Statistics.Minutes Delayed.Total'] / df_feat['Statistics.Flights.Total']
# Cancellation & diversion rates
df_feat['cancel_rate'] = df_feat['Statistics.Flights.Cancelled'] / df_feat['Statistics.Flights.Total']
df_feat['diversion_rate'] = df_feat['Statistics.Flights.Diverted'] / df_feat['Statistics.Flights.Total']
# One-hot encode month
month_dummies = pd.get_dummies(df_feat['Time.Month'], prefix='month', drop_first=True)
df_feat = pd.concat([df_feat, month_dummies], axis=1)

# Prepare feature matrix
feature_cols = list(delay_cols.keys()) + [
    'avg_delay_minutes', 'cancel_rate', 'diversion_rate'
] + list(month_dummies.columns)
X = df_feat[feature_cols].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Determine optimal k: Elbow Method
inertia = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
plt.figure()
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# 8. Determine optimal k: Silhouette Analysis
sil_scores = []
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))
plt.figure()
plt.plot(K, sil_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for k')
plt.show()

# 9. Fit final KMeans (choose k based on above; example k=4)
k_opt = 4
kmeans_final = KMeans(n_clusters=k_opt, random_state=42)
labels = kmeans_final.fit_predict(X_scaled)
df_feat['cluster'] = labels

# 10. Cluster profiling
cluster_profile = df_feat.groupby('cluster')[feature_cols].mean()
print("=== Cluster Profile (Mean Feature Values) ===")
print(cluster_profile)

# 11. PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
for c in range(k_opt):
    mask = labels == c
    plt.scatter(pcs[mask, 0], pcs[mask, 1], label=f'Cluster {c}', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection of Clusters')
plt.legend()
plt.tight_layout()
plt.show()