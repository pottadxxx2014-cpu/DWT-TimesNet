import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib


# =========================
# Nature-style helpers
# =========================
def _apply_nature_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.9, labelsize=11)
    ax.grid(False)


def _draw_break_marks(ax_high, ax_low, *, delta_y_data=0.003, d=0.015, color="k", lw=0.9):
    """Diagonal break marks on the left and right."""
    low_min, low_max = ax_low.get_ylim()
    high_min, high_max = ax_high.get_ylim()
    span_low = low_max - low_min
    span_high = high_max - high_min

    h_low = delta_y_data / span_low
    h_high = delta_y_data / span_high

    kwargs = dict(color=color, clip_on=False, linewidth=lw)

    kwargs["transform"] = ax_high.transAxes
    ax_high.plot((-d, +d), (-h_high, +h_high), **kwargs)

    kwargs["transform"] = ax_low.transAxes
    ax_low.plot((-d, +d), (1 - h_low, 1 + h_low), **kwargs)


# =========================
# Data loader
# =========================
def _compute_group_means(folder_list, metric_name):
    """Compute mean per method across result*.xls(x) files, per folder."""
    group_means = []

    for folder in folder_list:
        pattern_xlsx = os.path.join(folder, "result*.xlsx")
        pattern_xls = os.path.join(folder, "result*.xls")
        files = sorted(glob.glob(pattern_xlsx) + glob.glob(pattern_xls))

        if len(files) == 0:
            raise ValueError(f"No results files found in {folder}")

        method_vals = None

        for f in files:
            df = pd.read_excel(f)

            first_col = df.columns[0]
            df[first_col] = df[first_col].astype(str)

            rows = df[df[first_col] == metric_name]
            if rows.empty:
                raise ValueError(f"Metric '{metric_name}' not found in file: {f}")

            row = rows.iloc[0].drop(labels=[first_col])

            if method_vals is None:
                method_vals = {m: [] for m in row.index}

            for m in method_vals:
                method_vals[m].append(float(row[m]))

        method_mean = {m: float(np.mean(method_vals[m])) for m in method_vals}
        group_means.append(method_mean)

    methods = list(group_means[0].keys())
    return group_means, methods


# =========================
# Plotters
# =========================
def _plot_single_axis(ax, *, folder_list, metric_name, group_labels, colors, markers, title=None):
    group_means, methods = _compute_group_means(folder_list, metric_name)
    x = np.arange(1, len(folder_list) + 1)

    for i, method in enumerate(methods):
        y = [group_means[g][method] for g in range(len(folder_list))]
        ax.plot(
            x, y,
            color=colors[i],
            linewidth=1.8,
            marker=markers[i],
            markersize=5.8,
            markerfacecolor='white',
            markeredgewidth=1.4,
            markeredgecolor=colors[i],
            label=method
        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylim(0, 0.96)
    ax.set_yticks(np.arange(0, 1.0, 0.1))

    if title is not None:
        ax.set_title(title, fontsize=13, pad=6)

    _apply_nature_style(ax)
    return methods  # for legend


def _plot_broken_axis(
    ax_high,
    ax_low,
    *,
    folder_list,
    metric_name,
    group_labels,
    colors,
    y_break,
    title=None,
    markers,
    delta_y_data=0.003
):
    group_means, methods = _compute_group_means(folder_list, metric_name)
    x = np.arange(1, len(folder_list) + 1)

    (low_min, low_max), (high_min, high_max) = y_break

    for i, method in enumerate(methods):
        y = [group_means[g][method] for g in range(len(folder_list))]
        if any(not (low_min <= v <= low_max or high_min <= v <= high_max) for v in y):
            import warnings
            warnings.warn(method + '的结果超出原断轴显示范围，请调整原y_break参数，数据未被删除。')
        for a in (ax_high, ax_low):
            a.plot(
                x, y,
                color=colors[i],
                linewidth=1.8,
                marker=markers[i],
                markersize=5.8,
                markerfacecolor='white',
                markeredgewidth=1.4,
                markeredgecolor=colors[i],
                label=method
            )

    ax_low.set_ylim(low_min, low_max)
    ax_high.set_ylim(high_min, high_max)

    # bottom x only
    ax_low.set_xticks(x)
    ax_low.set_xticklabels(group_labels, fontsize=11)
    plt.setp(ax_high.get_xticklabels(), visible=False)
    ax_high.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # broken look
    ax_high.spines['bottom'].set_visible(False)
    ax_low.spines['top'].set_visible(False)

    _apply_nature_style(ax_high)
    _apply_nature_style(ax_low)

    _draw_break_marks(ax_high, ax_low, delta_y_data=delta_y_data, d=0.015, lw=0.9)

    if title is not None:
        ax_high.set_title(title, fontsize=13, pad=6)

    return methods  # for legend


def plot_wifi_ks_triptych(
    folder_list,
    *,
    group_labels=None,
    figsize=(9.8, 4.8),   # 👈 默认“更高一些、没那么宽”，你可改
    dpi=260,
    ylabel="KS statistic",
    legend_loc="lower center",
    legend_ncol=6,
):
    """
    Create a Nature-style 1×3 figure:
      - Peak timing (broken y)
      - Peak (broken y)
      - Variability factor (single y)
    One unified y-label, all axes keep y-ticks with 2 decimals.
    """
    if group_labels is None:
        group_labels = [str(i + 1) for i in range(len(folder_list))]

    _, actual_methods = _compute_group_means(folder_list, 'PeakTime-KS')
    color_map = {'Linear':'#4C6A92','Spline':'#5F7EA6','Markov':'#7B97B8',
                 'CNN':'#9EB6CF','Direct TimesNet':'#B7C8D8','DWT-TimesNet':'#C0362C'}
    marker_map = {'Linear':'o','Spline':'s','Markov':'D','CNN':'^','Direct TimesNet':'v','DWT-TimesNet':'h'}
    colors = [color_map[m] for m in actual_methods]
    markers = [marker_map[m] for m in actual_methods]

    # Settings (you can edit breaks/titles here)
    metrics = [
        dict(name="PeakTime-KS", title="Peak Timing", y_break=[[0, 0.11], [0.34, 0.47]]),
        dict(name="Peak-KS",     title="Peak Magnitude", y_break=[[0, 0.11], [0.31, 0.44]]),
        dict(name="VF-KS",       title="Volatility Factor", y_break=None),
    ]

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Outer: 2 rows (for broken axes) × 3 cols
    gs = fig.add_gridspec(
        2, 3,
        wspace=0.28,
        hspace=0.08
    )

    # Column 0 (broken)
    ax0_high = fig.add_subplot(gs[0, 0])
    ax0_low  = fig.add_subplot(gs[1, 0], sharex=ax0_high)

    # Column 1 (broken)
    ax1_high = fig.add_subplot(gs[0, 1])
    ax1_low  = fig.add_subplot(gs[1, 1], sharex=ax1_high)

    # Column 2 (single, span rows)
    ax2 = fig.add_subplot(gs[:, 2])

    # --- Plot ---
    methods0 = _plot_broken_axis(
        ax0_high, ax0_low,
        folder_list=folder_list,
        metric_name=metrics[0]["name"],
        group_labels=group_labels,
        colors=colors,
        y_break=metrics[0]["y_break"],
        title=metrics[0]["title"],
        markers=markers,
        delta_y_data=0.003
    )

    methods1 = _plot_broken_axis(
        ax1_high, ax1_low,
        folder_list=folder_list,
        metric_name=metrics[1]["name"],
        group_labels=group_labels,
        colors=colors,
        y_break=metrics[1]["y_break"],
        title=metrics[1]["title"],
        markers=markers,
        delta_y_data=0.003
    )

    methods2 = _plot_single_axis(
        ax2,
        folder_list=folder_list,
        metric_name=metrics[2]["name"],
        group_labels=group_labels,
        colors=colors,
        markers=markers,
        title=metrics[2]["title"],
    )

    # --- One unified y-label (left, centered) ---
    fig.supylabel(ylabel, fontsize=13, x=0.015)
    fig.supxlabel("Train Floor Numbers", fontsize=13, y=0.065)

    # --- Ensure ALL axes show y-ticks with 2 decimals ---
    fmt = mticker.FormatStrFormatter('%.2f')
    for ax in fig.axes:
        ax.yaxis.set_major_formatter(fmt)

    # --- Remove all per-axes legends ---
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # --- Bottom unified legend (use proxy handles) ---
    from matplotlib.lines import Line2D
    methods = methods2 or methods1 or methods0  # pick any non-empty list
    handles = [
        Line2D([0], [0], color=colors[i], lw=2,
               marker=markers[i],
               markersize=6,
               markerfacecolor='white',
               markeredgewidth=1.2,
               markeredgecolor=colors[i],
               label=methods[i])
        for i in range(min(len(methods), len(colors)))
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc=legend_loc,
            ncol=min(legend_ncol, len(handles)),
            frameon=False,
            fontsize=10.5,
            bbox_to_anchor=(0.5, -0.01)
        )

    # Layout: leave room for legend
    plt.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.16)

    plt.savefig('WIFI_data_analysis.png')
    plt.close(fig)

    # =========================================================
    all_rows = []

    # 先把每个指标的 group_means 都算好缓存起来
    metric_results = {}
    for m in metrics:
        group_means, methods = _compute_group_means(folder_list, m["name"])
        metric_results[m["title"]] = (group_means, methods)

    # 以第一个指标的方法列表为准（一般三个指标的方法列一致）
    methods = metric_results[metrics[0]["title"]][1]

    for gi, group in enumerate(group_labels):
        for method in methods:
            row = {
                "Group": group,      # 第一列：1/2/3/4
                "Method": method,    # 第二列：Linear/CNN/...
            }
            for m in metrics:
                title = m["title"]
                group_means = metric_results[title][0]
                row[title] = group_means[gi][method]
            all_rows.append(row)

    df_means = pd.DataFrame(all_rows)
    df_means.to_csv('WIFI_mean_metrics_V2_2.csv',index=False,encoding='utf-8-sig')

    # ======= 关键：打印全部，不截断 =======
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n=== Mean KS values (Group × Method) ===")
    print(df_means.to_string(index=False))

    return


# =========================
# Main entry
# =========================
def WIFI_analysis_main(file1, file2, file3, file4, *, figsize=(9.8, 4.8), dpi=260):
    folders = [file1, file2, file3, file4]

    matplotlib.rcParams['font.family'] = 'Calibri'

    plot_wifi_ks_triptych(
        folder_list=folders,
        figsize=figsize,
        dpi=dpi,
        ylabel="Mean KS Value",
    )
