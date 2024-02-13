import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster import hierarchy
import ast
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel("WCN20240203/result01.xlsx", sheet_name="result")
df = df.iloc[:, 1:]
df = df.T
df.columns = df.loc["Gnodes"]
df = df.iloc[1:, :]
columns = df.columns.to_list()
scaler = MinMaxScaler()

# # 对DataFrame中的数据进行0-1标准化
# df_scaled = scaler.fit_transform(df.T)
# df_scaled = pd.DataFrame(df_scaled.T, columns=df.columns)
df = df.T
df_scaled = (df - df.mean()) / df.std()
# df_scaled = (df - df.min()) / (df.max() - df.min())
df_scaled = df_scaled.T

spearman_corr = np.array(df_scaled.corr(method='pearson'))
spearman_corr_df = pd.DataFrame(spearman_corr, index=columns, columns=columns)

print(spearman_corr)
# 使用层次聚类，根据相似度重新排列数据
linkage_matrix = hierarchy.linkage(spearman_corr)
order = hierarchy.dendrogram(linkage_matrix, no_plot=True)['leaves']
spearman_corr_df = spearman_corr_df.iloc[order, order]
# 使用sns.clustermap创建聚类热力图
# plt.figure(figsize=(8, 10))
spearman_map = sns.clustermap(spearman_corr_df, cmap='coolwarm', annot=False, linewidths=0,
                              figsize=(8, 8),
                              tree_kws={'cmap': 'coolwarm'})  # Accent coolwarm BuGn
spearman_map.ax_heatmap.tick_params(axis='x', rotation=90)
spearman_map.ax_heatmap.tick_params(axis='y')

# plt.title('Spearman correlation', loc='center')
plt.savefig('WCN20240203/ClusterHeatmapSpearman.png')
plt.show()
