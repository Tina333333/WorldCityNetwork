import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('A new folder created.')
    else:
        print('Has already been created.')


def visualize_graph(graph, color0, size, corr):
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)
    cmap = matplotlib.colormaps.get_cmap("tab20")
    print(f'color is: {color0}')
    print(f'cmap is: {cmap}')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(color0))
    legend_labels_lower_left = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(sub_region_num)), markersize=8, alpha=0.8,
                   label=corr.get(sub_region_num)) for sub_region_num in list(set(color0))[:7]]
    legend_labels_lower_left2 = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(sub_region_num)), markersize=8, alpha=0.8,
                   label=corr.get(sub_region_num)) for sub_region_num in list(set(color0))[7:]]
    plt.title('Top 30 Cities')
    nx.draw_networkx(graph, with_labels=True, font_size=8, pos=nx.spring_layout(graph, seed=3210), width=0.08,
                     node_size=size, alpha=0.8, node_color=color0, cmap="tab20")
    plt.axis('off')
    plt.tight_layout()  # 缩小图形四周留白
    legend_up_left = ax.legend(handles=legend_labels_lower_left2, loc=(0.23, 0.016), ncol=1, fontsize=6)
    ax.add_artist(legend_up_left)
    ax.legend(handles=legend_labels_lower_left, loc="lower left", ncol=1, fontsize=6)
    plt.savefig(pwd + 'img/Top30Cities.png')
    plt.close()
    print("处理完成")


if __name__ == '__main__':
    pwd = './WCN20240203/'
    columns = []
    mkdir(pwd)

    df = pd.read_excel('top30cities.xlsx', 'Sheet1')
    head = df.head()
    print(f'head: {head}')
    print(f'shape: {df.shape}')

    color = pd.read_excel('top30cities.xlsx', 'Sheet2')
    head = color.head()
    print(f'head: {head}')
    print(f'shape: {color.shape}')
    print(f'Number: {color.Number}')
    color = color.Number

    sheet3 = pd.read_excel('top30cities.xlsx', sheet_name='Sheet3')
    _label = sheet3["No."]
    _region = sheet3["Sub-region"]
    _corr = {_label.loc[i]: _region.loc[i] for i in range(_label.shape[0])}

    # 构建无向图
    G = nx.from_pandas_edgelist(df, "Scity", "Dcity", edge_attr=True, create_using=nx.Graph)
    print(f'length of nodes: {len(G)}')
    de = dict(G.degree())  # 转换成dict 提出节点度
    # de2 = [de[v] * 5 for v in sorted(de.keys(), reverse=False)]
    de2 = [de[v] * 5 for v in de.keys()]
    print(f'degree: {de2}')

    nods = dict(G.nodes())
    print(f'nods: {nods}')
    visualize_graph(graph=G, color0=color, size=de2, corr=_corr)
