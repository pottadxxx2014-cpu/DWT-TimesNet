import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Sequence, Union
import os
import matplotlib
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


def _flatten_columns(cols: Union[pd.Index, pd.MultiIndex]) -> pd.Index:
    """把可能的多层列拍平成 'a-b-c' 形式。"""
    if isinstance(cols, pd.MultiIndex):
        return pd.Index(["-".join(map(str, t)) for t in cols.tolist()])
    return cols

def _col_suffixes(cols: Sequence[str]) -> list:
    """取列名最后一个 '-' 后面的后缀，用作 x 轴标签。"""
    return [str(c).split("-")[-1] for c in cols]

def _norm(s: str) -> str:
    """标准化行名/列名：去空格、统一大小写、把中文破折号等统一。"""
    return str(s).strip().replace("–", "-").replace("—", "-").replace(" ", "").lower()

# ---------- 核心：从多个 Excel 构建 3 张指标表 ----------
def build_metric_tables_from_files(
    file_paths: List[str],
    metric_rows: Sequence[str] = ("PeakTime-KS", "Peak-KS", "VF-KS", "HF-KS"),
    sheet: Union[int, str, None] = 0,
    index_labels: Sequence[str] = None
) -> Dict[str, pd.DataFrame]:
    if index_labels is None:
        index_labels = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]

    metric_tables = {m: [] for m in metric_rows}

    for name, path in zip(index_labels, file_paths):
        df = pd.read_excel(path, sheet_name=sheet, index_col=0)
        df.columns = _flatten_columns(df.columns)

        # 标准化 index，用于稳健定位行
        idx_map = {_norm(i): i for i in df.index}

        for m in metric_rows:
            key = _norm(m)
            if key not in idx_map:
                raise KeyError(f"[{name}] 中找不到行 '{m}'（标准化后 '{key}'），现有行包括：{list(df.index)[:15]} ...")

            real_row_name = idx_map[key]
            s = df.loc[real_row_name]
            s.name = name
            metric_tables[m].append(s)

    return {m: pd.DataFrame(metric_tables[m]) for m in metric_rows}

# ---------- 画图 ----------
def plot_metric_boxplots_sorted_together(
    metric_tables: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    rotation: int = 45,
    figsize=(9, 7),
    dpi=220,
    top_k=5,
    top_color="mediumvioletred",
    grey_color="#BDBDBD",
    grey_alpha=0.25
):
    """
    Nature-style boxplot
    """

    metrics_all = list(metric_tables.keys())
    metrics_for_sort = metrics_all
    metrics_for_plot = metrics_all[:-1]  # 不画最后一行

    # 共同列
    common_cols = None
    for m in metrics_for_sort:
        cols = metric_tables[m].columns
        common_cols = cols if common_cols is None else common_cols.intersection(cols)

    if common_cols is None or common_cols.empty:
        raise ValueError("各指标表没有共同列，请检查每个Excel的列名是否一致。")

    aligned = {m: metric_tables[m][common_cols] for m in metrics_for_sort}

    # ---------- 排序 ----------
    score = None
    for m in metrics_for_sort:
        w = weights.get(m, 1.0)
        med = aligned[m].median(axis=0, numeric_only=True)
        score = w * med if score is None else score + w * med

    ordered_cols = score.sort_values(ascending=True).index.tolist()
    xtick_labels = _col_suffixes(ordered_cols)

    # ---------- Top4 指标汇总成 DataFrame ----------
    top4 = ordered_cols[:4]

    rows = []

    for rank, col in enumerate(top4, start=1):
        row = {
            "Rank": rank,
            "Method": col
        }

        for m in metrics_for_sort:
            series = pd.to_numeric(aligned[m][col], errors="coerce")

            row[f"{m}_mean"] = series.mean()
            row[f"{m}_median"] = series.median()
            row[f"{m}_std"] = series.std()

        rows.append(row)

    top4_summary_df = pd.DataFrame(rows)

    # 更美观一点
    top4_summary_df = top4_summary_df.set_index(["Rank", "Method"])

    print("\nTop 4 Summary:")
    print(top4_summary_df.round(4))

    # def interleaved_4palettes_to_grey(n_cols: int,
    #                                   grey="#D0D0D0",
    #                                   blue="#1f77b4",
    #                                   red="#d62728",
    #                                   green="#2ca02c",
    #                                   orange="#ff7f0e"):
    #     """
    #     生成 n_cols 个颜色：
    #     先造4条渐变色盘(blue/red/green/orange -> grey)，每条长度=n_cols
    #     然后第j列取 palettes[j%4][j] 进行交错取色。
    #     """
    #     if n_cols <= 0:
    #         return []
    #
    #     base_colors = [blue, red, green, orange]
    #     grey_rgb = np.array(mcolors.to_rgb(grey))
    #
    #     palettes = []
    #     # 每条 palette：从 base -> grey 的线性插值，共 n_cols 个点
    #     t = np.linspace(0.0, 1.0, n_cols)  # 0鲜艳, 1灰
    #     for bc in base_colors:
    #         bc_rgb = np.array(mcolors.to_rgb(bc))
    #         pal = [tuple((1 - tt) * bc_rgb + tt * grey_rgb) for tt in t]
    #         palettes.append(pal)
    #
    #     # 交错取色：第j列取 palettes[j%4][j]
    #     colors = [palettes[j % 4][j] for j in range(n_cols)]
    #     return colors
    #
    # colors = interleaved_4palettes_to_grey(len(ordered_cols), grey="#D0D0D0")

    def mostly_blue_with_one_red_to_grey(
            n_cols: int,
            grey="#D0D0D0",
            blue_dark="#1F3A5F",
            blue_light="#7DAACF",
            red="#C0362C",
            red_index=1,  # 第二个
    ):
        """
        n_cols个颜色：大部分为蓝色(深->浅并向grey靠近)，仅第red_index个为红色系(红->grey)。
        """
        if n_cols <= 0:
            return []

        grey_rgb = np.array(mcolors.to_rgb(grey))
        b0 = np.array(mcolors.to_rgb(blue_dark))
        b1 = np.array(mcolors.to_rgb(blue_light))
        r0 = np.array(mcolors.to_rgb(red))

        # 蓝系：先做一个深->浅的蓝梯度，再逐步向grey收敛一点（更Nature）
        tb = np.linspace(0.0, 1.0, n_cols)
        blue_grad = [(1 - t) * b0 + t * b1 for t in tb]

        # 再往grey拉一点（控制“克制感”）
        fade = np.linspace(0.05, 0.45, n_cols)  # 可调：越大越灰
        blue_pal = [tuple((1 - f) * bg + f * grey_rgb) for bg, f in zip(blue_grad, fade)]

        # 红色：红->grey
        tr = 0.05  # 红色向灰靠近的程度（可调）
        red_col = tuple((1 - tr) * r0 + tr * grey_rgb)

        colors = blue_pal
        if 0 <= red_index < n_cols:
            colors[red_index] = red_col

        return colors

    colors = mostly_blue_with_one_red_to_grey(len(ordered_cols), grey="#D0D0D0", red_index=1)

    # ---------- 画图 ----------
    n = len(metrics_for_plot)
    fig, axes = plt.subplots(n, 1, figsize=figsize, dpi=dpi, sharex=True)

    if n == 1:
        axes = [axes]

    def _to_box_data(df, cols):
        return [pd.to_numeric(df[c], errors="coerce").dropna().values for c in cols]

    for i, m in enumerate(metrics_for_plot):
        ax = axes[i]
        data = _to_box_data(aligned[m], ordered_cols)

        bp = ax.boxplot(
            data,
            positions=np.arange(1, len(ordered_cols) + 1),
            widths=0.55,
            showfliers=False,
            patch_artist=False  # ❗ 关闭填充
        )

        for j in range(len(ordered_cols)):
            # box 边框
            bp["boxes"][j].set_color(colors[j])
            bp["boxes"][j].set_linewidth(1.2)

            # whiskers
            bp["whiskers"][2 * j].set_color(colors[j])
            bp["whiskers"][2 * j + 1].set_color(colors[j])

            # caps
            bp["caps"][2 * j].set_color(colors[j])
            bp["caps"][2 * j + 1].set_color(colors[j])

            # median
            bp["medians"][j].set_color(colors[j])
            bp["medians"][j].set_linewidth(1.4)

        # 平均值：黑边白心圆
        means = aligned[m].mean(axis=0, numeric_only=True).reindex(ordered_cols).values
        ax.plot(
            np.arange(1, len(ordered_cols) + 1),
            means,
            linestyle="",
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            zorder=3,
        )

        # Nature 风格：去网格
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)

        ylabel_map = {
            "PeakTime-KS": "Peak Timing KS",
            "Peak-KS": "Peak Magnitude KS",
            "VF-KS": "Volatility Factor KS",
        }

        ax.set_ylabel(ylabel_map.get(m, m), fontsize=12)
        ax.tick_params(axis="y", labelsize=11)

        if i != n - 1:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # 统一 x 轴
    axes[-1].set_xticks(np.arange(1, len(ordered_cols) + 1))
    axes[-1].set_xticklabels(
        xtick_labels,
        rotation=rotation,
        ha="center",
        fontsize=13
    )

    plt.tight_layout()

    # ---------- 贯穿三图的彩色背景带（每列一个颜色，alpha=0.20） ----------
    # 1) 减小子图间距，让“贯穿”更像一整块（可再调小/调大）
    fig.subplots_adjust(hspace=0.03)

    # 2) 先 draw 一下，确保 axes 位置已最终确定
    fig.canvas.draw()

    # 3) 让 axes 背景透明，否则会盖住 figure 背景带
    for ax in axes:
        ax.set_facecolor((1, 1, 1, 0))

    # 4) 取所有 axes 占据的 fig 纵向范围（从最底到最顶）
    pos_top = axes[0].get_position()
    pos_bot = axes[-1].get_position()
    y0 = pos_bot.y0
    y1 = pos_top.y1
    height = y1 - y0

    # 5) 用最下面的轴做 data->figure 坐标转换（x方向）
    ax_ref = axes[-1]

    for j in range(len(ordered_cols)):
        # 每列背景带覆盖 [j+1-0.5, j+1+0.5]
        x_left_data = (j + 1) - 0.5
        x_right_data = (j + 1) + 0.5

        # data -> display(px) -> figure 坐标
        x0_disp = ax_ref.transData.transform((x_left_data, 0))[0]
        x1_disp = ax_ref.transData.transform((x_right_data, 0))[0]
        x0_fig = fig.transFigure.inverted().transform((x0_disp, 0))[0]
        x1_fig = fig.transFigure.inverted().transform((x1_disp, 0))[0]
        width = x1_fig - x0_fig

        # 背景颜色：对应边框颜色 + 20%透明度
        r, g, b = mcolors.to_rgb(colors[j])
        face = (r, g, b, 0.05)

        rect = Rectangle(
            (x0_fig, y0), width, height,
            transform=fig.transFigure,
            facecolor=face,
            edgecolor="none",
            zorder=0.0
        )
        fig.add_artist(rect)

    plt.savefig('Different DWT analysis boxplots.png')

    plt.close(fig)
    return score.sort_values(ascending=True)


def DWT_analysis_main(folder_path):
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 200)
    matplotlib.rcParams['font.family'] = 'Calibri'

    # 读取目录下的所有 Excel
    file_list_choose = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.startswith('result_') and f.lower().endswith(('.xlsx', '.xls'))
    ]
    if not file_list_choose:
        raise FileNotFoundError(f"在目录 {folder_path} 中未找到任何 Excel 文件")

    metric_rows = ("PeakTime-KS", "Peak-KS", "VF-KS", "HF-KS")

    metric_tables = build_metric_tables_from_files(
        file_list_choose,
        metric_rows=metric_rows,
        index_labels=None
    )

    # 权重你自己定：例如更重视 VF/HF
    weights = {
        "PeakTime-KS": 1,
        "Peak-KS": 1,
        "VF-KS": 1,
        "HF-KS": 1
    }

    score_sorted = plot_metric_boxplots_sorted_together(
        metric_tables,
        weights=weights,
        rotation=-45,  # 45度
        figsize=(9, 7),  # 更紧凑
        dpi=220,  # 印刷级清晰度
        top_k=5  # 前5名高亮
    )

    return