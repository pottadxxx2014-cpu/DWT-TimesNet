
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, anderson_ksamp, cramervonmises_2samp
from scipy.spatial.distance import jensenshannon
import matplotlib
import os
from pathlib import Path
from datetime import datetime
import matplotlib.ticker as mticker


def compute_error_metrics(df):
    method_names = df.columns.tolist()[1:]

    truth = df.iloc[:, 0]
    mean_truth = np.mean(truth)

    prediction_list = []
    rmse_list = []
    cvmae_list = []
    ks_stat_list = []
    p_val_list = []

    for name in range(len(method_names)):
        model_prediction = df.iloc[:, name + 1]
        prediction_list.append(model_prediction)

        model_error = abs(model_prediction - truth)

        rmse = np.sqrt(np.mean(model_error ** 2))

        mae = np.mean(np.abs(model_error))
        cvmae = (mae / mean_truth * 100) if mean_truth != 0 else 0

        rmse_list.append(np.round(rmse, 4))
        cvmae_list.append(np.round(cvmae, 4))

        ks_stat, p_value = ks_2samp(truth, model_prediction)

        ks_stat_list.append(np.round(ks_stat, 4))
        p_val_list.append(np.round(p_value, 4))

    data = [rmse_list, ks_stat_list]
    df_metrics = pd.DataFrame(data, index=['RMSE', 'Occ-KS'], columns=method_names)

    # print('Error between ground truth data and generated data for indoor occupancy:')
    # print(df_metrics)

    return df_metrics


def _js_divergence_peak_time(truth_vals, pred_vals, n_bins=96, base=2):
    """
    计算峰值出现时刻的 JS divergence（不是 JS distance）。
    - truth_vals / pred_vals：每日峰值出现的时间索引（如 0..95），可包含 NaN
    - n_bins：离散类别数（15min 一天 96 个槽位；若用小时级则 n_bins=24）
    - base：对数底数（base=2 更常见）
    返回：
    - JS divergence（越小越好；0 表示分布完全一致）
    """
    t = pd.Series(truth_vals).dropna().astype(int).values
    p = pd.Series(pred_vals).dropna().astype(int).values

    if len(t) == 0 or len(p) == 0:
        return np.nan

    # 统计离散分布（直方图）
    t_counts = np.bincount(t, minlength=n_bins).astype(float)
    p_counts = np.bincount(p, minlength=n_bins).astype(float)

    if t_counts.sum() == 0 or p_counts.sum() == 0:
        return np.nan

    # 归一化成概率分布
    P = t_counts / t_counts.sum()
    Q = p_counts / p_counts.sum()

    # scipy 的 jensenshannon 返回 JS distance = sqrt(JS divergence)
    js_dist = jensenshannon(P, Q, base=base)
    return float(js_dist ** 2)


def js_permutation_test(truth_vals, pred_vals, n_bins=96, n_perm=2000, base=2, random_state=42):
    """
    JS divergence + 置换检验（Permutation Test），输出 JS 与 p-value。
    统计假设：
      H0：truth 与 pred 来自同一离散分布（峰值时刻分布一致）
      H1：两者分布不同
    检验方向：
      - JS 越大，差异越大
      - p-value 越小，说明差异越显著（不希望发生）
    因而“好”的表现通常是：JS 小 且 p >= 0.05

    返回：
      - js_obs：观测到的 JS divergence
      - p_value：置换检验得到的 p-value（右尾）
    """
    rng = np.random.default_rng(random_state)

    t = pd.Series(truth_vals).dropna().astype(int).values
    p = pd.Series(pred_vals).dropna().astype(int).values

    if len(t) == 0 or len(p) == 0:
        return np.nan, np.nan

    js_obs = _js_divergence_peak_time(t, p, n_bins=n_bins, base=base)

    # 合并样本，构造 H0 下的零分布：随机打乱后再分组
    z = np.concatenate([t, p])
    n_t = len(t)

    js_perm = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        rng.shuffle(z)
        t_star = z[:n_t]
        p_star = z[n_t:]
        js_perm[i] = _js_divergence_peak_time(t_star, p_star, n_bins=n_bins, base=base)

    # 右尾：在 H0 下，出现“>= 观测差异”的概率
    p_value = float(np.mean(js_perm >= js_obs))
    return float(js_obs), p_value


def _ad_2samp_stat_p(truth_vals, pred_vals):
    """
    Anderson–Darling 两样本检验（尾部敏感），用于峰值大小分布。
    返回：
      - ad_stat：AD 统计量（越小通常越接近）
      - ad_p：近似 p-value（由 significance_level(%) / 100 得到，越小越显著不同）
    解释：
      ad_p < 0.05：说明峰值分布与真实存在显著差异（尤其尾部）
      ad_p >= 0.05：无法拒绝 H0，可认为峰值分布一致性更好（你想要）
    """
    x = pd.Series(truth_vals).dropna().astype(float).values
    y = pd.Series(pred_vals).dropna().astype(float).values

    if len(x) < 5 or len(y) < 5:
        return np.nan, np.nan

    res = anderson_ksamp([x, y])
    ad_stat = float(res.statistic)

    # significance_level 通常以“百分数”给出（越小越显著）
    ad_p = float(res.significance_level) / 100.0
    return ad_stat, ad_p


def _cvm_2samp_stat(truth_vals, pred_vals):
    """
    Cramér–von Mises 两样本统计量：
    - 用于连续变量分布的整体差异度量
    - statistic 越小越接近真实分布
    """
    x = pd.Series(truth_vals).dropna().astype(float).values
    y = pd.Series(pred_vals).dropna().astype(float).values

    if len(x) < 5 or len(y) < 5:
        return np.nan, np.nan

    res = cramervonmises_2samp(x, y)
    res = ks_2samp(x, y)
    return float(res.statistic), float(res.pvalue)


def compute_daily_metrics(df):
    """
    输出（日尺度）：
      - peak_time_df: 每日峰值出现时刻（0~points_per_day-1），若多峰取第一个最大值位置
      - max_df: 每日峰值
      - VF_df: 每日VF（|diff|之和 / 日均值）
      - HF_df: 每日高频能量占比 HF_ratio（频率>f0 的能量 / 总能量，排除DC）
    """
    points_per_day = 96
    dt_hours = 0.25  # 采样率15min
    f0 = 1.0  # 1小时以内定义为高频
    detrend_mean = True

    day_num = len(df.index) // points_per_day

    def _daily_hf_ratio(day_data):
        x = np.asarray(day_data, dtype=float)
        if detrend_mean:
            x = x - np.mean(x)

        N = len(x)
        if N == 0:
            return 0.0

        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(N, d=dt_hours)  # cycles/hour
        S = np.abs(X) ** 2

        # 排除 DC
        mask_total = freqs > 0
        total_energy = S[mask_total].sum()
        if total_energy <= 0:
            return 0.0

        mask_hf = freqs > f0
        hf_energy = S[mask_hf].sum()

        return float(hf_energy / total_energy)

    def compute_daily_metrics_for_col(input_df, col_name, days):
        max_list = []
        peak_time_list = []
        VF_list = []
        HF_list = []

        for i in range(days):
            day_data = input_df[col_name].iloc[i * points_per_day:(i + 1) * points_per_day].to_numpy()

            # Max
            day_max = float(np.max(day_data)) if len(day_data) > 0 else np.nan

            peak_pos = (
                int(np.where(day_data == np.nanmax(day_data))[0][-1])
                if len(day_data) > 0 else np.nan
            )

            # VF
            abs_diff_sum = float(np.abs(np.diff(day_data)).sum()) if len(day_data) > 1 else 0.0
            day_avg = float(np.mean(day_data)) if len(day_data) > 0 else 0.0
            diff_metric = abs_diff_sum / day_avg if day_avg != 0 else 0.0

            # HF ratio
            hf_ratio = _daily_hf_ratio(day_data)

            max_list.append(day_max)
            peak_time_list.append(peak_pos)
            VF_list.append(diff_metric)
            HF_list.append(hf_ratio)

        return max_list, peak_time_list, VF_list, HF_list

    max_dict, peak_time_dict, VF_dict, HF_dict = {}, {}, {}, {}

    method_names = df.columns.tolist()
    for col in method_names:
        occ_max_list, occ_peak_time_list, occ_VF_list, occ_HF_list = compute_daily_metrics_for_col(df, col, day_num)
        max_dict[col] = occ_max_list
        peak_time_dict[col] = occ_peak_time_list
        VF_dict[col] = occ_VF_list
        HF_dict[col] = occ_HF_list

    max_df = pd.DataFrame(max_dict)
    peak_time_df = pd.DataFrame(peak_time_dict)
    VF_df = pd.DataFrame(VF_dict)
    HF_df = pd.DataFrame(HF_dict)

    return peak_time_df, max_df, VF_df, HF_df


def metrics_picture(
    peak_t_df, max_df, VF_df, HF_df,
    peak_bins=96,            # 峰值时刻离散类别数：15min=96；小时级=24
    js_perm_n=5000,          # JS 置换检验次数（越大越稳，但越慢）
    js_random_state=42,      # 固定随机种子，保证可复现
    round_n=4
):
    models = [c for c in max_df.columns if c != "Truth"]

    js_list, js_p_list = [], []
    peak_stat_list, peak_p_list = [], []
    vf_cvm_list, vf_p_list = [], []
    hf_cvm_list, hf_p_list = [], []

    for m in models:
        # ---------- 1) 峰值出现时刻：JS + p-value（置换检验） ----------
        # js_obs, js_p = js_permutation_test(
        #     truth_vals=peak_t_df["Truth"],
        #     pred_vals=peak_t_df[m],
        #     n_bins=peak_bins,
        #     n_perm=js_perm_n,
        #     base=2,
        #     random_state=js_random_state
        # )
        # js_list.append(np.round(js_obs, round_n))
        # js_p_list.append(np.round(js_p, round_n))

        js_stat, js_p = _cvm_2samp_stat(peak_t_df["Truth"], peak_t_df[m])
        js_list.append(np.round(js_stat, round_n))
        js_p_list.append(np.round(js_p, round_n))

        # ---------- 2) 峰值大小：Peak 的 CvM ----------
        peak_stat, peak_p = _cvm_2samp_stat(max_df["Truth"], max_df[m])
        peak_stat_list.append(np.round(peak_stat, round_n))
        peak_p_list.append(np.round(peak_p, round_n))

        # ---------- 3) 时域波动性：VF 的 CvM ----------
        vf_cvm, vf_p = _cvm_2samp_stat(VF_df["Truth"], VF_df[m])
        vf_cvm_list.append(np.round(vf_cvm, round_n))
        vf_p_list.append(np.round(vf_p, round_n))

        # ---------- 4) 频域波动性：HF_ratio 的 CvM ----------
        hf_cvm, hf_p = _cvm_2samp_stat(HF_df["Truth"], HF_df[m])
        hf_cvm_list.append(np.round(hf_cvm, round_n))
        hf_p_list.append(np.round(hf_p, round_n))

    # 汇总为 “行=指标，列=模型” 的结果表
    out = pd.DataFrame(
        [js_list, js_p_list, peak_stat_list, peak_p_list, vf_cvm_list, vf_p_list, hf_cvm_list, hf_p_list],
        index=["PeakTime-KS", "PeakTime-P", "Peak-KS", "Peak-P", "VF-KS", "VF-P", "HF-KS", "HF-P"],
        columns=models
    )
    return out


def random_picture(df, n_days=10, out_prefix="Date", dpi=300):
    """
    Nature/AE-like time-series style (reference-code style):
      - Truth: dark grey solid line
      - Linear: dashed + hollow circle markers (color #4C6A92)
      - TimesNet (DWT-TimesNet): dashed + hollow pentagon markers (color #C0362C)
      - No confidence interval band
      - Remove top/right spines, ticks outward, light (optional) y-grid
    """
    matplotlib.rcParams['font.family'] = 'Calibri'
    # --- prepare index to date/time columns ---
    df = df.copy()
    df.index = df.index.astype(str)
    df = df.reset_index().rename(columns={'index': 'datetime'})
    df['date'] = df['datetime'].apply(lambda x: x.split()[0])
    df['time'] = df['datetime'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else '')

    unique_dates = df['date'].unique()
    if len(unique_dates) == 0:
        return

    n_days = min(n_days, len(unique_dates))
    selected_dates = random.sample(list(unique_dates), n_days)

    # --- fixed scheme requested by your colleague ---
    col_name = ['Truth', 'Linear', 'DWT-TimesNet']  # 如果你列名是 TimesNet，请在这里改一下
    colors = {
        'Truth': '0.20',        # 深灰
        'Linear': '#4C6A92',    # 指定蓝
        'DWT-TimesNet': '#C0362C'  # 指定红
    }
    markers = {
        'Truth': None,
        'Linear': 'o',
        'DWT-TimesNet': 'h'
    }

    for d in selected_dates:
        day_df = df[df['date'] == d].reset_index(drop=True)

        # x 用序号更稳，tick 再映射回 time 字符串（和你之前做法一致）
        x = np.arange(len(day_df))

        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)

        # ===== 1) Ground Truth =====
        ax.plot(
            x,
            pd.to_numeric(day_df[col_name[0]], errors='coerce').values,
            label='Ground Truth',
            linewidth=2.5,
            color=colors['Truth'],
            alpha=0.5
        )

        # ===== 2) Linear =====
        ax.plot(
            x,
            pd.to_numeric(day_df[col_name[1]], errors='coerce').values,
            label='Linear',
            linestyle='--',
            linewidth=0.8,
            color=colors['Linear'],
            alpha=0.95,
            marker=markers['Linear'],
            markersize=3.0,
            markerfacecolor='white',
            markeredgewidth=0.6,
            markeredgecolor=colors['Linear'],
        )

        # ===== 3) TimesNet / DWT-TimesNet =====
        ax.plot(
            x,
            pd.to_numeric(day_df[col_name[2]], errors='coerce').values,
            label='DWT-TimesNet',
            linestyle='--',
            linewidth=0.8,
            color=colors['DWT-TimesNet'],
            alpha=0.95,
            marker=markers['DWT-TimesNet'],
            markersize=3.2,
            markerfacecolor='white',
            markeredgewidth=0.6,
            markeredgecolor=colors['DWT-TimesNet'],
        )

        # ===== title =====
        ax.set_title(f"{d}", fontsize=13, pad=6)

        ymax = np.nanmax([
            pd.to_numeric(day_df[col_name[0]], errors='coerce').max(),
            pd.to_numeric(day_df[col_name[1]], errors='coerce').max(),
            pd.to_numeric(day_df[col_name[2]], errors='coerce').max()
        ])

        if ymax <= 0: ymax = 1.0  # 全零日仅保护坐标，不改数据
        ax.set_ylim(-ymax * 0.025, ymax * 1.05)

        # 只显示整数
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Occupancy Number", fontsize=12)

        # ===== ticks: every 4 points (15-min data -> hourly) =====
        tick_indices = [i for i in range(len(day_df)) if (i + 1) % 4 == 0]
        tick_labels = [day_df['time'].iloc[i] for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=10)

        ax.tick_params(axis='y', labelsize=10, direction='out', length=4, width=0.8)
        ax.tick_params(axis='x', direction='out', length=4, width=0.8)

        # ===== spines =====
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.9)
        ax.spines['bottom'].set_linewidth(0.9)

        # ===== grid (如果你想完全 Nature 极简，改成 ax.grid(False)) =====
        ax.grid(axis='y', linestyle='--', linewidth=0.4, color='0.9', alpha=0.8)
        ax.set_axisbelow(True)

        # ===== legend =====
        ax.legend(
            fontsize=10,
            frameon=False,
            loc='upper left',
            handlelength=3
        )

        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{d}_occupancy.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return


def plot_offset_histogram(df, compare_name, x_label_name, model_col, color, bins_set):
    # ==============================
    # Nature-style global settings
    # ==============================
    matplotlib.rcParams.update({
        "font.family": "Calibri",
        "font.size": 13,
        "axes.linewidth": 1.0,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon": False,
        "legend.fontsize": 12,
        "figure.dpi": 300
    })

    # ==============================
    # 数据处理（保持你原逻辑）
    # ==============================
    truth = df['Truth']
    Q3 = np.percentile(truth, 75)
    truth_max = truth.max()
    upper_bound = min(truth_max * 1.1, 2.5 * Q3)
    if upper_bound <= 0: upper_bound = max(float(truth_max), 1.0)

    bins = np.linspace(0, upper_bound, bins_set + 1)
    bin_width = bins[1] - bins[0]

    truth_counts, _ = np.histogram(truth, bins=bins)
    model_counts, _ = np.histogram(df[model_col], bins=bins)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    offset = bin_width / 5
    truth_x = bin_centers - offset
    model_x = bin_centers + offset

    # ==============================
    # 绘图
    # ==============================
    fig, ax = plt.subplots(figsize=(8, 5))

    # Model
    ax.bar(
        model_x, model_counts,
        width=bin_width / 2.5,
        align='center',
        color=color,
        alpha=0.85,
        label=model_col
    )

    # Truth（更克制的浅色）
    ax.bar(
        truth_x, truth_counts,
        width=bin_width / 2.5,
        align='center',
        color='#D6EAF8',  # 更高级浅蓝
        edgecolor='black',
        linewidth=0.6,
        alpha=0.9,
        label='Truth'
    )

    # ==============================
    # Nature-style轴设置
    # ==============================
    ax.set_ylabel("Frequency")
    ax.set_xlabel(x_label_name)
    ax.set_xlim(-0.02 * upper_bound, upper_bound)

    # 去掉上右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 轴线细化
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # 不加网格（Nature通常无网格）
    ax.grid(False)

    # Legend更紧凑
    ax.legend(loc="upper right")

    # 更好的留白
    plt.tight_layout()

    # 保存
    plt.savefig(
        f"Frequency Distribution - Truth vs {model_col} in {compare_name}.png")

    plt.close('all')


def result_analysis(result_df, folder_path_set, picture_set, seed_set, cross_set, log_dir=None, additional_set=False):
    print('\nStart Model Validation:')
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)

    # 检验逐时刻误差、日人数分布、日最大人数分布、日人数VF因子分布
    error_df = compute_error_metrics(result_df)
    peak_t_df, max_df, VF_df, HF_df = compute_daily_metrics(result_df)
    total_statistic_P_df = metrics_picture(peak_t_df, max_df, VF_df, HF_df)
    combined_results = pd.concat([error_df, total_statistic_P_df], axis=0)
    print('Error metrics and Daily metrics between ground truth data and generated data for indoor occupancy:')
    print(combined_results)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        combined_results.to_csv(Path(log_dir)/'main_metrics.csv', encoding='utf-8-sig')
        result_df.to_csv(Path(log_dir)/'predictions.csv', encoding='utf-8-sig')

    # 绘制图像
    color_name_old = ['lightblue', 'lightcoral', 'mediumseagreen', 'goldenrod', 'dodgerblue', 'teal', 'tomato', 'mediumvioletred', 'mediumorchid', 'crimson']
    color_name = ['#6E6E6E', '#4C6A92', '#5F7EA6', '#7B97B8', '#9EB6CF', '#C0362C']
    model_name = ['Truth', 'Linear', 'Spline','Markov', 'CNN', 'DWT-TimesNet']
    x_label_names = ['Daily Occupancy Peak Timing', 'Daily Occupancy Peak Magnitude', 'Daily Occupancy Volatility Factor', 'Daily Occupancy HF Value']
    if picture_set:
        for model_c in range(1, len(model_name)):
            if model_name[model_c] == 'DWT-TimesNet':
                plot_offset_histogram(peak_t_df, 'Peak_Time', x_label_names[0], model_name[model_c], color_name[model_c], 25)
                plot_offset_histogram(max_df, 'Peak_Value',x_label_names[1], model_name[model_c], color_name[model_c], 25)
                plot_offset_histogram(VF_df, 'VF',x_label_names[2], model_name[model_c], color_name[model_c], 25)
                plot_offset_histogram(HF_df, 'HF', x_label_names[3], model_name[model_c], color_name[model_c], 25)

        # 随机绘制一日内逐时刻人数
        random_picture(result_df)

    # 创建文件夹路径
    folder_path = folder_path_set

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if cross_set:
        file_path = f'{folder_path}/result_seed{seed_set}_{time_str}.xlsx'
        combined_results.to_excel(file_path, index=True)
    else:
        file_path = f'{folder_path}/result_{time_str}.xlsx'
        raw_path = f'{folder_path}/generate_data_{time_str}.xlsx'
        combined_results.to_excel(file_path, index=True)
        result_df.to_excel(raw_path, index=True)

    if log_dir is not None:
        import json
        from runtime_analysis import runtime_summary
        (Path(log_dir)/'result_files.json').write_text(
            json.dumps({'main_result':str(Path(file_path).resolve())},ensure_ascii=False,indent=2),encoding='utf-8')
        runtime_summary(log_dir)
        if additional_set:
            from additional_analysis import additional_results_analysis
            additional_results_analysis(result_df,log_dir,picture_set=picture_set,postprocess_set=not cross_set)
    return combined_results

