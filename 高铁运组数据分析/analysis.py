import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from pyecharts import options as opts
from pyecharts.charts import Map
from utils import colors, updated_station_to_city


def plot_one(data):
    df = data.parse(sheet_name="流量流向")
    df.set_index("日期", inplace=True)
    month_data = df.loc["月计"]
    stations = month_data.index.tolist()[:-1]
    flows = month_data.values.tolist()[:-1]
    d = defaultdict(int)
    for i, station in enumerate(stations):
        city = updated_station_to_city[station]
        d[city] += int(flows[i])
    perc = np.percentile(list(d.values()), np.arange(0, 110, 10))
    pieces = []
    for i in range(len(perc) - 1):
        pieces.append({
            "min": perc[i],
            "max": perc[i + 1],
            "label": f"{int(perc[i])} - {int(perc[i + 1])}",
            "color": colors[i]
        })
    c = (
        Map()
        .add(
            "商家A",
            [list(z) for z in zip(d.keys(), d.values())],
            "china-cities",
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="30天深圳北站总流量流向示意图"),
            visualmap_opts=opts.VisualMapOpts(is_piecewise=True, pieces=pieces,
                                              range_text=["", "流量范围(人次)"]),
            legend_opts=opts.LegendOpts(is_show=False)
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .render(r"results\流量流向地理图.html")
    )


# 各车次的供给与需求关系示意图
def plot_two(data, sheets):
    def prepare_data(d, text):
        d = d[d.index.notna()].iloc[:-3]  # drop null rows
        d = d.dropna(axis="columns", how="all")  # drop null columns
        d = d.loc[:, d.columns[d.columns.get_loc(text):]].dropna()
        d.columns = ["定员", "无座", "合计", "发送", "上座率"]
        d["日均定员"], d["日均发送"] = d["定员"] / 10, d["发送"] / 10
        d = d.apply(pd.to_numeric, errors='coerce')
        return d
    # 处理上/中/下旬数据
    first_ten = prepare_data(data.parse(sheet_name=sheets[0], index_col="车次"), text="上旬")
    sec_ten = prepare_data(data.parse(sheet_name=sheets[1], index_col="车次"), text="中旬")
    thr_ten = prepare_data(data.parse(sheet_name=sheets[2], index_col="车次"), text="下旬")
    # 处理整月数据
    sheet_four = data.parse(sheet_name=sheets[3], index_col="车次")  # parse sheet one
    sheet_four = sheet_four[sheet_four.index.notna()].iloc[:-4]
    common_index = first_ten.index.intersection(sec_ten.index).intersection(thr_ten.index)
    all_month = sheet_four.loc[:, sheet_four.columns[sheet_four.columns.get_loc("月计"):]].dropna()
    all_month.columns = ["定员", "无座", "合计", "发送", "上座率"]
    all_month["日均定员"], all_month["日均发送"] = all_month["定员"] / 30, all_month["发送"] / 30
    all_month = all_month.apply(pd.to_numeric, errors='coerce')
    all_month = all_month[all_month.index.isin(common_index)]
    # 绘图
    con_data = [[first_ten, sec_ten], [thr_ten, all_month]]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    titles = ["上旬各车次供需关系图", "中旬各车次供需关系图", "下旬各车次供需关系图", "全月各车次供需关系图"]
    for i in range(2):
        for j in range(2):
            mean_capacity = con_data[i][j]['日均定员'].mean()
            mean_passenger = con_data[i][j]['日均发送'].mean()
            sns.regplot(x="日均定员", y="日均发送", data=con_data[i][j], ax=axes[i][j], scatter_kws={'s': 8})
            # 绘制平均值点
            axes[i][j].plot(mean_capacity, mean_passenger, 'ro')
            axes[i][j].set_xlabel('列车定员', fontsize=14)
            axes[i][j].set_ylabel('日均发送', fontsize=14)
            axes[i][j].set_title(titles[2 * i + j], fontsize=16)
            axes[i][j].plot(np.arange(1250))
            axes[i][j].annotate(f"平均发送人数及定员={int(mean_capacity), int(mean_passenger)}",
                                xy=(mean_capacity, mean_passenger + 30),
                                xytext=(mean_capacity - 100, mean_passenger + 300),
                                arrowprops=dict(arrowstyle='->',color='blue', lw=1,
                                                connectionstyle='arc3,rad=-0.5'),
                                horizontalalignment="right", verticalalignment="top")
            axes[i][j].axhline(y=mean_passenger, color='black', linestyle='--')
            axes[i][j].axvline(x=mean_capacity, color='black', linestyle='--')
    fig.subplots_adjust(wspace=0.12)
    plt.savefig(fname=r"results\供给需求关系示意图.png", dpi=400)


# 列车开点与上座率的关系示意图
def plot_three(data, sheets):
    def prepare_data(d, start, end):
        d = d[d.index.notna()].iloc[:-3].dropna(axis="columns", how="all")
        d = pd.merge(d["开点"], d.iloc[:, 1::5], left_index=True, right_index=True)
        d = d.dropna()
        cols = ["开点", "到站"]
        cols.extend(map(str, list(range(start, end))))
        cols.append("上旬平均")
        d.columns = cols
        d['开点'] = pd.to_datetime(d['开点'], format='%H:%M:%S').dt.strftime('%H')
        for col in d.columns[2:]:
            d[col] = d[col].astype(float)
        d.set_index("开点", inplace=True)
        d = d.iloc[:, 1:]
        d = d.groupby('开点').mean()
        return d
    data_one = prepare_data(data.parse(sheet_name=sheets[0], index_col="车次"), 1, 11)
    data_two = prepare_data(data.parse(sheet_name=sheets[1], index_col="车次"), 11, 21)
    data_three = prepare_data(data.parse(sheet_name=sheets[2], index_col="车次"), 21, 31)
    data_all = data.parse(sheet_name=sheets[3], index_col="车次")
    data_all = data_all[data_all.index.notna()].iloc[:-4].dropna(axis="columns", how="all")
    data_all = pd.merge(data_all["开点"], data_all.iloc[:, -1], left_index=True, right_index=True)
    data_all = data_all.dropna()
    data_all.columns = ["开点", "月总"]
    data_all['开点'] = pd.to_datetime(data_all['开点'], format='%H:%M:%S').dt.strftime('%H')
    data_all["月总"] = data_all["月总"].astype(float)
    data_all.set_index("开点", inplace=True)
    data_all = data_all.groupby('开点')['月总'].mean()
    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(25, 35))
    data_list = [[data_one, data_two], [data_three, data_all]]
    title_list = [["上旬各日", "中旬各日"], ["下旬各日", "月平均"]]
    for i in range(2):
        for j in range(2):
            data_list[i][j].plot.bar(ax=axes[i * 2 + j], rot=0, width=0.8)
            axes[i * 2 + j].set_xlabel('开点', fontsize=20)
            axes[i * 2 + j].set_ylabel('上座率', fontsize=20)
            axes[i * 2 + j].tick_params(axis='both', which='major', labelsize=20)
            axes[i * 2 + j].set_title(f"{title_list[i][j]}车次开点和上座率关系", fontsize=30)
    plt.savefig(fname=r"results\列车开点与上座率关系示意图.png", dpi=400)


def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    data = pd.ExcelFile("raw_data.xls")
    sheets = data.sheet_names
    plot_one(data)
    plot_two(data, sheets)
    plot_three(data, sheets)


if __name__ == '__main__':
    main()
