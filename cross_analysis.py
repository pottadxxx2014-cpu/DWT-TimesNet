
import os
import json
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as m_color
import matplotlib
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def cross_analysis_function(file_list):
    print('\nSummary Data in Cross Validation:')

    first_df = None
    numeric_df_list = []

    for file in file_list:
        df = pd.read_excel(file, index_col=[0])

        if first_df is None:
            first_df = df.copy()

        df_numeric = df.apply(pd.to_numeric, errors='coerce')

        # 保证所有文件的 index 和 columns 顺序一致
        df_numeric = df_numeric.reindex(index=first_df.index, columns=first_df.columns)

        numeric_df_list.append(df_numeric)

    # shape: [n_files, n_rows, n_cols]
    data_array = np.stack(
        [df.values.astype(float) for df in numeric_df_list],
        axis=0
    )

    mean_array = np.nanmean(data_array, axis=0)
    median_array = np.nanmedian(data_array, axis=0)
    q1_array = np.nanquantile(data_array, 0.25, axis=0)
    q3_array = np.nanquantile(data_array, 0.75, axis=0)

    def make_stat_df(stat_array):
        stat_df = first_df.copy()
        stat_df.loc[:, :] = np.round(stat_array, 4)
        return stat_df

    mean_df = make_stat_df(mean_array)
    median_df = make_stat_df(median_array)
    q1_df = make_stat_df(q1_array)
    q3_df = make_stat_df(q3_array)

    print('\nMean Data:')
    print(mean_df)

    print('\nMedian Data:')
    print(median_df)

    print('\nQ1 Data:')
    print(q1_df)

    print('\nQ3 Data:')
    print(q3_df)

    output_file = 'Summary Data in Cross Validation.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        mean_df.to_excel(writer, sheet_name='Mean')
        median_df.to_excel(writer, sheet_name='Median')
        q1_df.to_excel(writer, sheet_name='Q1')
        q3_df.to_excel(writer, sheet_name='Q3')

    print(f'\nSaved to: {output_file}')

    return mean_df, median_df, q1_df, q3_df


def ks_data_fetch_fun(file_list):
    frames = [pd.read_excel(file, index_col=0) for file in file_list]
    names = ['PeakTime-KS', 'Peak-KS', 'VF-KS', 'HF-KS']
    tables = []
    for name in names:
        if any(name not in frame.index for frame in frames): raise KeyError('结果缺少指标：' + name)
        table = pd.DataFrame([frame.loc[name] for frame in frames])
        table.index = [f'{name}_{i+1}' for i in range(len(table))]
        tables.append(table)
    return tables, names


def violin_plot_drawing(df_choose, name_choose, color_list, model_list, writer_excel):
    # 仅修改画图显示名称，不修改数据列名
    display_model_list = [
        'TimesNet' if name == 'Direct TimesNet' else name
        for name in model_list
    ]

    # ====== 1) 统计量输出到 Excel（25/mean/50/75） ======
    stats = pd.DataFrame({
        '25%': df_choose.quantile(0.25),
        'mean': df_choose.mean(),
        '50%': df_choose.quantile(0.50),
        '75%': df_choose.quantile(0.75)
    })
    stats.T.to_excel(writer_excel, sheet_name=str(name_choose)[:31], float_format='%.4f')

    # ====== 2) 画图：rain + box + half-violin + mean-X ======
    fig, ax = plt.subplots(figsize=(12, 8))

    n_methods = df_choose.shape[1]

    for j, col in enumerate(df_choose.columns):
        y = pd.to_numeric(df_choose[col], errors='coerce').dropna().values
        c = color_list[j]

        # (1) 左侧 scatter（rain）
        x_scatter = np.random.normal(loc=j - 0.25, scale=0.04, size=len(y))
        ax.scatter(
            x_scatter, y,
            s=30,
            alpha=0.9,
            color='white',
            edgecolor=c,
            linewidth=1.2,
            zorder=2
        )

        # (2) 中间 boxplot（对齐 xtick）
        bp = ax.boxplot(
            y,
            positions=[j],
            widths=0.18,
            patch_artist=True,
            showfliers=False,
        )
        for box in bp['boxes']:
            box.set(facecolor='white', edgecolor='black', linewidth=2)
        for median in bp['medians']:
            median.set(color='black', linewidth=2)
        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=1.5)
        for cap in bp['caps']:
            cap.set(color='black', linewidth=1.5)

        # (3) 右半边 violin（cloud）
        parts = ax.violinplot(
            y,
            positions=[j],
            widths=0.5,
            showmeans=False,
            showmedians=False,
            showextrema=False
        ) if len(y) > 1 and np.ptp(y) > 0 else {'bodies': []}
        for pc in parts['bodies']:
            verts = pc.get_paths()[0].vertices
            x_center = np.mean(verts[:, 0])
            verts[:, 0] = np.maximum(verts[:, 0], x_center)  # 只保留右半边

            pc.set_facecolor(c)
            pc.set_edgecolor('none')
            pc.set_alpha(0.6)
            pc.set_zorder(1.5)

    # (4) mean 用 X 标记（放在 box 中心位置）
    means = stats['mean'].values
    for j, mean in enumerate(means):
        ax.plot(
            j, mean,
            marker='o',
            color='white',
            markersize=8,
            markeredgecolor='black',
            markeredgewidth=1.5,
            zorder=3.5
        )

    # ====== 3) y 轴范围：沿用你原来的逻辑 ======
    if name_choose == 'PeakTime-KS':
        ax.set_ylim([min(df_choose.min()) - 0.03, max(df_choose.max()) + 0.03])
        name_set = 'Peak Timing'
        y_step = 0.04
    elif name_choose == 'Peak-KS':
        ax.set_ylim([min(df_choose.min()) - 0.015, max(df_choose.max()) + 0.015])
        name_set = 'Peak Magnitude'
        y_step = 0.02
    elif name_choose == 'VF-KS':
        ax.set_ylim([min(df_choose.min()) - 0.05, max(df_choose.max()) + 0.05])
        name_set = 'Volatility Factor'
        y_step = 0.1
    elif name_choose == 'HF-KS':
        ax.set_ylim([min(df_choose.min()) - 0.06, max(df_choose.max()) + 0.06])
        name_set = 'HF Value'
        y_step = 0.1
    else:
        ax.set_ylim([min(df_choose.min()) - 0.003, max(df_choose.max()) + 0.003])
        name_set = 'XXX Value'
        y_step = 0.02

    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # ====== 4) 轴、标题、xticks ======
    ax.set_ylabel(f'KS for {name_set}', fontsize=20)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(display_model_list, fontsize=20, rotation=-30)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', length=4, width=1.0)

    # ====== 5) Nature-style：去掉 top/right + 淡网格 ======
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    # ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.22)
    ax.set_axisbelow(True)

    # ====== 6) 图例放右侧中部 ======
    legend_elements = [
        plt.Line2D([0], [0], marker='o', linestyle='',
                   markersize=10, markerfacecolor=c, markeredgecolor='none',
                   label=name)
        for c, name in zip(color_list, display_model_list)
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=18,
        frameon=False,
        bbox_to_anchor=(0.98, 0.5),
        loc='center left',
        borderaxespad=0.
    )

    plt.tight_layout()
    plt.savefig(f'Distribution of {name_choose} for all Models.png', dpi=300)
    plt.close('all')
    return


def rank_pickup_fun(df_choose, name_choose):
    ranked_df = df_choose.apply(lambda row: row.rank(method='min', ascending=True), axis=1)
    rank_total = len(df_choose.columns)
    # Step 2: 统计每一列排名的次数
    rank_count_df = pd.DataFrame(0, index=[f'rank{i}' for i in range(1, rank_total + 1)], columns=df_choose.columns)

    # 计算每一列的排名次数
    for col in ranked_df.columns:
        for i in range(1, rank_total + 1):  # 对于1到8的每个排名
            rank_count_df.loc[f'rank{i}', col] = (ranked_df[col] == i).sum()

    # print(f'{name_choose}\n', rank_count_df)
    # 绘制堆积柱状图
    col_totals = rank_count_df.sum(axis=0)
    rank_percentage_df = rank_count_df / col_totals * 100
    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = m_color.LinearSegmentedColormap.from_list('color', ['mediumorchid', 'lightgrey'])

    # 绘制堆积柱状图，使用渐变颜色
    rank_percentage_df.T.plot(kind='bar', stacked=True, ax=ax, colormap=cmap, width=0.6)

    plt.title(f'Stacked Bar Chart of Rank Percentages in {name_choose}', fontsize=20)
    plt.ylabel('Rank Percentages (%)', fontsize=20)

    plt.xticks(rotation=-30, fontsize=18)
    plt.yticks(fontsize=14)
    plt.subplots_adjust(left=0.1, right=0.8, bottom=0.15)
    plt.legend(bbox_to_anchor=(1.025, 0.5), loc='center left', fontsize=20)
    plt.savefig(f'Rank Percentages for {name_choose}.png')
    plt.close('all')

    return


def print_pvalue_ratio_last_method(file_list, p_row_indices=(3, 5, 7, 9)):
    # 参数保留兼容原调用；实际按名称读取，避免行号错误。
    frames = [pd.read_excel(file, index_col=0) for file in file_list]
    print('\n===== Descriptive P-value > 0.05 ratio (not distributional equivalence) =====')
    for metric in ['PeakTime-P', 'Peak-P', 'VF-P', 'HF-P']:
        values = np.asarray([f.loc[metric, 'DWT-TimesNet'] for f in frames], dtype=float)
        values = values[np.isfinite(values)]
        if len(values): print(f'{metric}: {(values>0.05).mean()*100:.2f}% ({(values>0.05).sum()}/{len(values)})')
    return


def cross_analysis_function_main(folder_path, outpath):
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    matplotlib.rcParams['font.family'] = 'Calibri'

    # 读取数据
    file_list_choose = completed_result_files(folder_path)

    if not file_list_choose:
        raise FileNotFoundError(f"在目录 {folder_path} 中未找到任何 Excel 文件")

    print_pvalue_ratio_last_method(file_list_choose, p_row_indices=(3, 5, 7, 9))

    # 求平均值
    cross_analysis_function(file_list_choose)

    # 获取不同指标的统计表格
    df_list, name_list = ks_data_fetch_fun(file_list_choose)

    # 绘制小提琴图
    model_name = list(df_list[0].columns)
    color_map = {'Linear':'#4C6A92', 'Spline':'#5F7EA6', 'Markov':'#7B97B8',
                 'CNN':'#9EB6CF', 'Direct TimesNet':'#B7C8D8', 'DWT-TimesNet':'#C0362C'}
    color_name = [color_map.get(name, '#888888') for name in model_name]

    with pd.ExcelWriter(outpath) as writer:
        for df_choose0, name_choose0 in zip(df_list, name_list):
            violin_plot_drawing(df_choose0, name_choose0, color_name, model_name, writer)

    # # 制作排名表格
    # for df_choose0, name_choose0 in zip(df_list, name_list):
    #     rank_pickup_fun(df_choose0, name_choose0)

    summarize_additional_runs(folder_path)
    return

def completed_result_files(folder_path):
    root = Path(folder_path)
    runs = root / 'training_runs'
    files = []
    seen = set()
    if runs.exists():
        for done in sorted(runs.glob('cross_seed*/completed.json')):
            folder = done.parent
            settings = json.loads((folder/'run_settings.json').read_text(encoding='utf-8'))
            seed = int(settings['seed'])
            if seed in seen:
                raise ValueError(f'seed={seed}有多份完整结果，请先分开不同实验配置，不自动择优/重复计数。')
            ref = json.loads((folder/'result_files.json').read_text(encoding='utf-8'))
            basename = ref['main_result'].replace('\\','/').split('/')[-1]
            path = root / basename
            if not path.exists(): raise FileNotFoundError(path)
            seen.add(seed); files.append(str(path))
    else:
        for path in sorted(root.glob('result_seed*.xlsx')):
            match = re.search(r'result_seed(\d+)_',path.name)
            if match:
                seed = int(match.group(1))
                if seed in seen: raise ValueError(f'seed={seed}重复，不自动选择结果。')
                seen.add(seed)
            files.append(str(path))
    if files:
        cols=[list(pd.read_excel(f,index_col=0).columns) for f in files]
        if any(c != cols[0] for c in cols): raise ValueError('结果包含不同模型列，不能混合五模型与六模型。')
    return files


def summarize_additional_runs(folder_path):
    records=[]
    runtime_records=[]
    for done in sorted((Path(folder_path)/'training_runs').glob('cross_seed*/completed.json')):
        folder=done.parent
        settings=json.loads((folder/'run_settings.json').read_text(encoding='utf-8'))
        runtime_file=folder/'runtime_comparison.csv'
        if runtime_file.exists():
            runtime=pd.read_csv(runtime_file);runtime['seed']=int(settings['seed']);runtime_records.append(runtime)
        source=folder/'additional'/'additional_metrics.csv'
        if not source.exists():continue
        frame=pd.read_csv(source,index_col=0)
        frame['seed']=int(settings['seed']);frame['Model']=frame.index
        records.append(frame.reset_index(drop=True))
    if not records:return
    all_rows=pd.concat(records,ignore_index=True)
    if all_rows.duplicated(['seed','Model']).any():raise ValueError('附加汇总存在重复seed/model。')
    out=Path(folder_path)/'additional_summary';out.mkdir(exist_ok=True)
    all_rows.to_csv(out/'all_runs.csv',index=False,encoding='utf-8-sig')
    with pd.ExcelWriter(out/'additional_summary.xlsx') as writer:
        all_rows.to_excel(writer,sheet_name='All runs',index=False)
        for label,method in [('Mean','mean'),('Median','median')]:
            getattr(all_rows.drop(columns='seed').groupby('Model'),method)(numeric_only=True).to_excel(writer,sheet_name=label)
    if runtime_records:
        timings=pd.concat(runtime_records,ignore_index=True)
        timings.to_csv(out/'runtime_all_runs.csv',index=False,encoding='utf-8-sig')
        with pd.ExcelWriter(out/'runtime_summary.xlsx') as writer:
            timings.to_excel(writer,sheet_name='All runs',index=False)
            timings.drop(columns='seed').groupby('Model').median(numeric_only=True).to_excel(writer,sheet_name='Median')
    print('附加指标汇总：',out/'additional_summary.xlsx')
