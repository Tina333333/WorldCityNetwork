from pyecharts.charts import *
from pyecharts import options as opts
import pandas as pd
from pyecharts.commons.utils import JsCode
# 导入输出图片工具
from pyecharts.render import make_snapshot
# 使用snapshot_selenium渲染图片
from snapshot_selenium import snapshot

airports = pd.read_excel('airports_dat.xlsx')
routes = pd. read_excel('routes_dat.xlsx')


# 绘制全球机场散点图
def airports_viz(airports_data):
    """
    绘制全球机场散点图
    :param airports_data: 机场位置数据
    :return: pyecharts GEO图表实例
    """
    data_pair = []
    geo = Geo(init_opts=opts.InitOpts(theme='white', bg_color='white', width='990px', height='540px'))
    # bg_color='#000000', width='1980px', height='1080px'
    for tup in zip(airports_data['id'], airports_data['longitude'], airports_data['latitude']):
        # 新增机场位置信息到geo
        geo.add_coordinate(str(tup[0]), tup[1], tup[2])
        data_pair.append([str(tup[0]), 1])

    geo.add_schema(
        maptype="world",
        is_roam=False,  # 禁止缩放
        zoom=1.1,  # 显示比例
        itemstyle_opts=opts.ItemStyleOpts(color="#D3D3D3", border_color="#1E90FF"),
        # color="#000000", border_color="#1E90FF"
        emphasis_label_opts=opts.LabelOpts(is_show=False),
        emphasis_itemstyle_opts=opts.ItemStyleOpts(color="#323c48")
    )

    geo.add("Airports",
            data_pair,
            type_='scatter',
            # is_selected=True,
            symbol_size=3,
            is_large=True,
            itemstyle_opts=opts.ItemStyleOpts(color="#DC143C")
            # color="#E1FFFF"
            )

    print('The form of airports is ', data_pair[10])
    # 关闭Label显示
    geo.set_series_opts(label_opts=opts.LabelOpts(is_show=False))

    geo.set_global_opts(
        title_opts=opts.TitleOpts(title="Global Airports", pos_top='3%', pos_left='center'),
        tooltip_opts=opts.TooltipOpts(is_show=False),  # 关闭提示框
        legend_opts=opts.LegendOpts(is_show=False, pos_left='left', pos_bottom='4', orient='vertical'),
        graphic_opts=[
            opts.GraphicGroup(
                graphic_item=opts.GraphicItem(
                    # 控制整体的位置
                    left="left",
                    bottom="bottom",
                ),
                children=[
                    # opts.GraphicText控制文字的显示
                    opts.GraphicText(
                        graphic_item=opts.GraphicItem(
                            left="center",
                            top="middle",
                            z=100,
                        ),
                        graphic_textstyle_opts=opts.GraphicTextStyleOpts(
                            # 可以通过jsCode添加js代码，也可以直接用字符串
                            text=JsCode(
                                "['Python 3.10.10 ', 'pyecharts 2.0.3 ',"
                                "'https://github.com/Tina333333/WorldCityNetwork'].join('\n')"
                            ),
                            # font="14px Microsoft YaHei",
                            graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(
                                fill="#333"
                            )
                        )
                    )
                ]
            )
        ],
    )

    return geo


def flights_line_viz(airports_data, routes_data):
    geo = Geo(init_opts=opts.InitOpts(theme='white', bg_color='white', width='990px', height='540px'))
    # bg_color='#000000', width='1980px', height='1080px'
    # 添加机场的坐标点
    for tup in zip(airports_data['id'],  airports_data['longitude'], airports_data['latitude']):
        geo.add_coordinate("airports" + str(tup[0]), tup[1], tup[2])

    geo.add_schema(
        maptype="world",
        is_roam=False,
        zoom=1.1,
        itemstyle_opts=opts.ItemStyleOpts(color="#D3D3D3", border_color="#1E90FF"),
        # color="#000000, , border_color="#1E90FF"
        emphasis_label_opts=opts.LabelOpts(is_show=False),
        emphasis_itemstyle_opts=opts.ItemStyleOpts(color="#323c48")
    )

    data_pair = []
    for tup in zip(routes_data['Source airport ID'], routes_data['Destination airport ID']):
        data_pair.append(["airports" + str(tup[0]), "airports" + str(tup[1])])
    geo.add("Routes",
            data_pair,
            type_='lines',
            # is_selected=True if airlines[idx][0] == 'Air China' else False,
            symbol_size=0.5,
            is_large=True,
            large_threshold=1e6,
            progressive_threshold=100000,
            linestyle_opts=opts.LineStyleOpts(curve=0.2, opacity=0.03, color='#426F42', width=0.7),
            # color='#1E90FF'
            )
    print('The form of routes is ', data_pair[0])
    geo.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 关闭标签显示

    geo.set_global_opts(
        title_opts=opts.TitleOpts(title="Global Air Routes", pos_top='3%', pos_left='center'),
        tooltip_opts=opts.TooltipOpts(is_show=False),   # 关闭提示框
        legend_opts=opts.LegendOpts(is_show=False, pos_left='left', pos_top='2', orient='vertical'),
        graphic_opts=[
            opts.GraphicGroup(
                graphic_item=opts.GraphicItem(
                    # 控制整体的位置
                    left="left",
                    bottom="bottom",
                ),
                children=[
                    # opts.GraphicText控制文字的显示
                    opts.GraphicText(
                        graphic_item=opts.GraphicItem(
                            left="center",
                            top="middle",
                            z=100,
                        ),
                        graphic_textstyle_opts=opts.GraphicTextStyleOpts(
                            # 可以通过jsCode添加js代码，也可以直接用字符串
                            text=JsCode(
                                "['Python 3.10.10 ', 'pyecharts 2.0.3 ',"
                                "'https://github.com/Tina333333/WorldCityNetwork'].join('\n')"
                            ),
                            # font="14px Microsoft YaHei",
                            graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(
                                fill="#333"
                            )
                        )
                    )
                ]
            )
        ],
    )

    return geo


if __name__ == "__main__":
    charts = airports_viz(airports)
    charts.render('airports.html')
    make_snapshot(snapshot, charts.render(), 'airports.png', delay=8)
    make_snapshot(snapshot, charts.render(), 'airports.pdf', delay=8)

    charts = flights_line_viz(airports, routes)
    charts.render('routes.html')
    make_snapshot(snapshot, charts.render(), 'routes.png', delay=100)
    make_snapshot(snapshot, charts.render(), 'routes.pdf', delay=100)
