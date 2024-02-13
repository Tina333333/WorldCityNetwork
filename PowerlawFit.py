import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import math
import powerlaw
import numpy as np
import os
import matplotlib


def save_csv(list1, columns0, csv_filename):
    _dict = {}
    for index, _c in enumerate(columns0):
        _dict.update({_c: list1[index]})
        print(_dict)
        print(len(list1[index]))
    df0 = pd.DataFrame(_dict)
    df0.to_csv(csv_filename, index=False)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('A new folder created.')
    else:
        print('Has already been created.')


def loglog0(ds, data, alph, gamm):
    hist, bin_edges = np.histogram(data, bins=100)
    hist = hist / n
    m = len(bin_edges)
    impo = bin_edges[1:m]
    estiprob = (impo ** (-alph)) * (np.e ** (-gamm * impo)) * (bin_edges[1] - bin_edges[0])
    estiprob = estiprob / np.sum(estiprob)
    print(f'the sum of estiprob is: {np.sum(estiprob)}')
    plt.loglog(impo, hist, 'o', color='g', label='samples')  # marker .,o^s,square
    plt.loglog(impo[0:m - 20], estiprob[0:m - 20], linestyle='-', color='r', label='generalized random graph')
    # marker .,o^s,square
    plt.legend()
    plt.xlabel(ds + ' $x$')
    plt.ylabel("Probability $P(x)$")
    plt.savefig(pwd + 'img/' + ds + 'scatter.png', bbox_inches='tight')
    plt.close()


def gcn00(adj_e, d_half, x):
    x1 = d_half @ x
    x1 = adj_e @ x1
    x1 = d_half @ x1
    sum0 = np.sum(x1)
    x1 = x1 / sum0
    return x1


def visualize_graph(graph, color, size, corr):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cmap = matplotlib.colormaps.get_cmap("tab20")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(color))
    legend_labels_up_left = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(sub_region_num)), markersize=10,
                   label=corr.get(sub_region_num)) for sub_region_num in list(set(color))[:6]]

    legend_labels_lower_left = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(sub_region_num)), markersize=10,
                   label=corr.get(sub_region_num)) for sub_region_num in list(set(color))[6:12]]

    legend_labels_lower_right = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(sub_region_num)), markersize=10,
                   label=corr.get(sub_region_num)) for sub_region_num in list(set(color))[12:]]

    plt.title('World City Network')
    nx.draw_networkx(graph, with_labels=False, font_size=8, pos=nx.spring_layout(graph, seed=3210), width=0.08,
                     node_size=size, alpha=0.8, node_color=color, cmap="tab20")
    plt.axis('off')
    legend_up_left = ax.legend(handles=legend_labels_up_left, loc="upper left", ncol=1)
    ax.add_artist(legend_up_left)
    legend_up_right = ax.legend(handles=legend_labels_lower_right, loc="lower right", ncol=1)
    ax.add_artist(legend_up_right)
    ax.legend(handles=legend_labels_lower_left, loc="lower left", ncol=1)

    plt.savefig(pwd + './img/AllCities.png')
    plt.savefig(pwd + './img/AllCities.pdf')
    plt.close()
    print("图片处理完成")


def aic_weights(a, b, c, d):
    # AIC赤池信息准则
    aic_min = min([a, b, c, d])
    delt_a = a - aic_min
    delt_b = b - aic_min
    delt_c = c - aic_min
    delt_d = d - aic_min

    sum_all = sum([math.exp(-delt_a / 2), math.exp(-delt_b / 2), math.exp(-delt_c / 2), math.exp(-delt_d / 2)])
    aic_a = math.exp(-delt_a / 2) / sum_all
    aic_b = math.exp(-delt_b / 2) / sum_all
    aic_c = math.exp(-delt_c / 2) / sum_all
    aic_d = math.exp(-delt_d / 2) / sum_all

    print(f'aic_a is: {aic_a: .4f}')
    print(f'aic_b is: {aic_b: .4f}')
    print(f'aic_c is: {aic_c: .4f}')
    print(f'aic_d is: {aic_d: .4f}')


def powerlaw0_fit(ds, data):
    print("======================================================")
    print(ds)
    fita = powerlaw.Fit(data)
    aic_exp = -2 * (sum(fita.exponential.loglikelihoods(data))) + 1 * 2
    aic_pow = -2 * (sum(fita.power_law.loglikelihoods(data))) + 1 * 2
    aic_log = -2 * (sum(fita.lognormal.loglikelihoods(data))) + 2 * 2
    aic_tru = -2 * (sum(fita.truncated_power_law.loglikelihoods(data))) + 2 * 2

    fig_pdf = fita.plot_pdf(linestyle='None', marker='o', color='g', label='samples')
    fita.lognormal.plot_pdf(color='b', linestyle='--', label='lognormal', ax=fig_pdf)
    fita.exponential.plot_pdf(color='c', linestyle='--', label='exponential', ax=fig_pdf)
    fita.power_law.plot_pdf(color='m', linestyle='--', label='power law', ax=fig_pdf)
    fita.truncated_power_law.plot_pdf(color='r', linestyle='-', label='truncated power law', ax=fig_pdf)

    fig_pdf.set_ylabel('Frequency')
    fig_pdf.set_xlabel('x')
    handles, labels = fig_pdf.get_legend_handles_labels()
    fig_pdf.legend(handles, labels, loc=3, prop={'size': 8})
    plt.savefig(pwd + 'img/' + ds + 'powerlaw.png', bbox_inches='tight')
    plt.clf()
    aic_weights(aic_exp, aic_pow, aic_log, aic_tru)

    print(f'fita.exponential.parameter1 is: {fita.exponential.parameter1: .4f}')
    print(f'fita.power_law.xmin is is: {fita.power_law.xmin: .4f}')
    print(f'fita.power_law.alpha is: {fita.power_law.alpha: .4f}')
    print(f'fita.lognormal.mu is: {fita.lognormal.mu: .4f}')
    print(f'fita.lognormal.sigma is: {fita.lognormal.sigma: .4f}')
    print(f'fita.truncated_power_law.alpha is: {fita.truncated_power_law.alpha: .4f}')
    print(f'fita.truncated_power_law.xmin is: {fita.truncated_power_law.xmin: .4f}')
    print(f'fita.truncated_power_law.parameter2: {fita.truncated_power_law.parameter2}')

    return fita.truncated_power_law.alpha, fita.truncated_power_law.parameter2


if __name__ == '__main__':
    pwd = './WCN20240203/'
    mkdir(pwd)

    df = pd.read_excel('CityLinesMinus.xlsx', sheet_name='Sheet1')
    head = df.head()
    print(f'head: {head}')
    print(f'shape: {df.shape}')

    world_cities = []
    f = open("./world_city_20231127.txt", "r")
    for line in f.readlines():
        linestr = line.strip()
        linestrlist = linestr.split("\t")
        world_cities.append(linestrlist[1] + linestrlist[2])

    subregion = pd.read_excel('CityLinesMinus.xlsx', sheet_name='Sheet2')
    head = subregion.head()
    print(f'head: {head}')
    print(f'shape: {subregion.shape}')

    sheet5 = pd.read_excel('CityLinesMinus.xlsx', sheet_name='Sheet5')
    _label = sheet5["No."]
    _region = sheet5["Sub-region"]
    _corr = {_label.loc[i]: _region.loc[i] for i in range(_label.shape[0])}

    # 构建无向图
    G = nx.from_pandas_edgelist(df, "SCitySCountry", "DCityDCountry", edge_attr=True, create_using=nx.Graph)
    # 增加节点分区属性
    nx.set_node_attributes(G, subregion.set_index('SCitySCountry').to_dict('index'))
    n = len(G)

    # 转换成dict 提出节点度
    de = dict(G.degree())
    de2 = [de[v] * 5 for v in de.keys()]

    nods = list(G.nodes.data('Label'))
    print(f'nods[0:10]: {nods[0:10]}')
    # 转换成dict 节点的颜色或者区域属性
    color1 = dict(G.nodes.data('Label'))
    color2 = [color1[u] for u in color1.keys()]
    # print(f'color2: {color2}')
    # 做所有城市的网络图
    visualize_graph(G, color=color2, size=de2, corr=_corr)

    # 绘制重要性、拟合对数函数
    adj = nx.to_numpy_array(G)  # 老版本用nx.to_numpy_matrix(G)
    print(f'adjacent matrix: {adj}')
    eye = np.eye(n)
    adj_eye = adj + eye
    print(f'adjacent and eye matrix: {adj_eye}')
    de3 = [(de[v] + 1) ** (-0.5) for v in de.keys()]
    D_half = np.diag(de3)
    # Dhalf = np.linalg.matrix_power(Dhalf, 2)
    print(f'D_half: {D_half}')
    H0 = np.ones((n, 1))
    print(f'H0 :{H0}')
    for i in range(500):
        H1 = gcn00(adj_e=adj_eye, d_half=D_half, x=H0)
        H0 = H1
        if i % 50 == 0:
            print(f'the i is: {i}')
            print(f'the H0 is: {H0}')
    H3 = np.array(H0)
    GCNrank = H3[:, 0]
    GCNrank = list(GCNrank)

    columns = []
    DeBePaGC = []

    Gnodes = list(G.nodes)
    DeBePaGC.append(Gnodes)
    columns.append('Gnodes')

    degree0 = dict(G.degree)
    degree0 = [degree0[v] for v in degree0.keys()]
    DeBePaGC.append(degree0)
    columns.append("Degree")
    alpha, gamma = powerlaw0_fit('Degree', degree0)
    loglog0('Degree', degree0, alpha, gamma)

    betweenness = nx.betweenness_centrality(G)
    betweenness = [1000*betweenness[v] for v in betweenness.keys()]
    DeBePaGC.append(betweenness)
    columns.append("Betweenness")
    # 去零操作
    # betweenness = [value for value in betweenness if value > 0.5]
    alpha, gamma = powerlaw0_fit('Betweenness', betweenness)
    loglog0('Betweenness', betweenness, alpha, gamma)

    pagerank = nx.pagerank(G)
    pagerank = [1000*pagerank[v] for v in pagerank.keys()]
    alpha, gamma = powerlaw0_fit('Pagerank', pagerank)
    loglog0('Pagerank', pagerank, alpha, gamma)
    DeBePaGC.append(pagerank)
    columns.append("Pagerank")

    GCNrank = [1000*value for value in GCNrank]
    DeBePaGC.append(GCNrank)
    columns.append("GCNrank")
    alpha, gamma = powerlaw0_fit('GCNrank', GCNrank)
    loglog0('GCNrank', GCNrank, alpha, gamma)

    save_csv(DeBePaGC, columns, pwd + 'DeBePaGC.csv')
    df = pd.read_csv(pwd + 'DeBePaGC.csv')
    result = df[df['Gnodes'].isin(world_cities)]
    result.to_csv(pwd + 'result.csv', index=False)

    # closeness_centrality = nx.closeness_centrality(G)
    # closeness_centrality = [closeness_centrality[v] for v in closeness_centrality.keys()]
    # print(f'closeness_centrality: {closeness_centrality}')
    # powerlaw0_fit(closeness_centrality)

    # de_c = nx.degree_centrality(G)
    # de_c = [de_c[v] for v in de_c.keys()]
    # alpha, gamma = powerlaw0_fit(de_c)
    # loglog0('Degree centrality', de_c, alpha, gamma)
