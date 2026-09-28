"""保留V2_1附加指标口径，供V2_2的additional_analysis.py复用。
也可独立运行本文件，只分析最终generate_data文件；不训练或修改原主表。
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from model_validation import compute_daily_metrics


CONFIG = {
    'folder_path': './Results_V2_2_Raw',
    'prediction_file': None,  # None读取该文件夹中时间戳最新的generate_data_*.xlsx
    'outfolder': './Results_V2_2_Raw/additional',
    'acf_lags': [1, 2, 4, 8, 24, 48, 96],  # 单位：15分钟；各周独立计算
}


def _mae(x):
    return float(np.mean(np.abs(x)))


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def _finite_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def check_prediction_data(result_df):
    """完整保留预测顺序，不排序、不删除缺失行、不填补非法值。"""
    if not result_df.columns.is_unique or result_df.columns[0] != 'Truth':
        raise ValueError('第一列必须是Truth，模型列名不能重复。请读取generate_data文件，不是result指标表。')
    if len(result_df.columns) < 2:
        raise ValueError('结果至少应包含Truth和一个预测列。')
    if len(result_df) == 0 or len(result_df) % 672:
        raise ValueError('V2_1按完整周生成结果，预测行数必须是672的整数倍。')
    values = result_df.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('发现NaN或无穷大；请先核查预测，不能通过删行破坏整点或日期对齐。')
    # 原始标签是“日期 0:15”至“日期 24:00”，支持补零的小时写法。
    actual_times = []
    for label in result_df.index:
        time_text = str(label).rsplit(' ', 1)[-1]
        try:
            hours, minutes = (int(v) for v in time_text.split(':'))
        except (ValueError, TypeError):
            raise ValueError('时间标签应为“日期 小时:分钟”；无法确认日内位置：' + str(label))
        actual_times.append(hours * 60 + minutes)
    expected = np.tile(np.arange(1, 97) * 15, len(result_df) // 96)
    if not np.array_equal(actual_times, expected):
        raise ValueError('时间标签不是按日0:15至24:00排列，请先核查数据，不能按错误位置计算非整点误差。')
    return result_df.astype(float)


def point_metrics_generate(result_df):
    truth = result_df['Truth'].to_numpy(dtype=float)
    position = np.arange(len(result_df)) % 4
    unknown = position != 3
    rows = []
    for name in result_df.columns[1:]:
        pred = result_df[name].to_numpy(dtype=float)
        error = pred - truth
        row = {
            'Model': name,
            'MAE_all': _mae(error), 'RMSE_all': _rmse(error),
            'MAE_unknown': _mae(error[unknown]),
            'RMSE_unknown': _rmse(error[unknown]),
            'MAE_hourly_anchor': _mae(error[~unknown]),
            'MaxAbsError_hourly_anchor': float(np.max(np.abs(error[~unknown]))),
            'Negative_points': int(np.sum(pred < 0)),
            'Noninteger_points': int(np.sum(np.abs(pred - np.round(pred)) > 1e-8)),
        }
        for offset, minute in enumerate([15, 30, 45]):
            selected = error[position == offset]
            row[f'MAE_{minute}min'] = _mae(selected)
            row[f'RMSE_{minute}min'] = _rmse(selected)
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')


def daily_metrics_generate(result_df):
    # 原VF公式、HF公式及末次峰值规则直接复用，不悄悄改原评价定义。
    peak, max_value, vf, hf = compute_daily_metrics(result_df)
    truth_days = result_df['Truth'].to_numpy(dtype=float).reshape(-1, 96)
    true_mean = truth_days.mean(axis=1)
    day_names = [str(v).rsplit(' ', 1)[0] for v in result_df.index[::96]]
    paired_tables = []
    summary_rows = []
    for name in result_df.columns[1:]:
        pred_days = result_df[name].to_numpy(dtype=float).reshape(-1, 96)
        available = (max_value['Truth'].to_numpy() > 0) & (max_value[name].to_numpy() > 0)
        peak_error = (peak[name].to_numpy() - peak['Truth'].to_numpy()) * 15
        peak_abs_valid = np.where(available, np.abs(peak_error), np.nan)
        ramp_error = np.diff(pred_days, axis=1) - np.diff(truth_days, axis=1)
        day_error = pred_days - truth_days
        vf_error = vf[name].to_numpy() - vf['Truth'].to_numpy()
        hf_error = hf[name].to_numpy() - hf['Truth'].to_numpy()
        peak_value_error = max_value[name].to_numpy() - max_value['Truth'].to_numpy()
        paired_tables.append(pd.DataFrame({
            'Model': name, 'Day_order': np.arange(1, len(day_names) + 1),
            'Week_order_in_output': np.arange(len(day_names)) // 7 + 1,
            'Date_label': day_names,
            'MAE_unknown': np.abs(day_error[:, np.arange(96) % 4 != 3]).mean(axis=1),
            'RMSE_unknown': np.sqrt(np.square(day_error[:, np.arange(96) % 4 != 3]).mean(axis=1)),
            'Peak_truth': max_value['Truth'].to_numpy(), 'Peak_pred': max_value[name].to_numpy(),
            'Peak_value_error': peak_value_error,
            'Peak_time_error_min_last': peak_error,
            'Peak_time_abs_error_valid_min': peak_abs_valid,
            'Both_have_positive_peak': available,
            'VF_truth': vf['Truth'].to_numpy(), 'VF_pred': vf[name].to_numpy(),
            'VF_abs_error': np.abs(vf_error),
            'VF_truth_mean_zero': true_mean == 0,
            'VF_pred_mean_zero': pred_days.mean(axis=1) == 0,
            'HF_truth': hf['Truth'].to_numpy(), 'HF_pred': hf[name].to_numpy(),
            'Ramp_MAE_per_15min_step': np.abs(ramp_error).mean(axis=1),
        }))
        summary_rows.append({
            'Model': name, 'N_days': len(day_names),
            'Peak_value_MAE': _mae(peak_value_error),
            'Peak_time_MAE_valid_min_last': _finite_mean(peak_abs_valid),
            'N_days_valid_peak_time': int(available.sum()),
            'N_days_excluded_peak_time': int((~available).sum()),
            'VF_MAE_all_legacy_zero_rule': _mae(vf_error),
            'VF_bias_all_legacy_zero_rule': float(vf_error.mean()),
            'VF_KS_all_legacy_zero_rule': float(ks_2samp(vf['Truth'], vf[name]).statistic),
            'HF_MAE': _mae(hf_error),
            'Ramp_MAE_per_15min_step': _mae(ramp_error),
            'N_truth_zero_mean_days': int(np.sum(true_mean == 0)),
            'N_pred_zero_mean_days': int(np.sum(pred_days.mean(axis=1) == 0)),
        })
    return pd.DataFrame(summary_rows).set_index('Model'), pd.concat(paired_tables, ignore_index=True)


def _acf_one_week(values, lag):
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        return np.nan  # 常数周ACF未定义，后面显式计数。
    return float(np.dot(centered[:-lag], centered[lag:]) / denominator)


def acf_metrics_generate(result_df, lags):
    if not lags or any(not isinstance(v, (int, np.integer)) or not 0 < v < 672 for v in lags):
        raise ValueError('acf_lags应为1至671的非空整数列表。')
    truth_weeks = result_df['Truth'].to_numpy(dtype=float).reshape(-1, 672)
    records = []
    summary_rows = []
    for name in result_df.columns[1:]:
        pred_weeks = result_df[name].to_numpy(dtype=float).reshape(-1, 672)
        errors = []
        for week, (truth, pred) in enumerate(zip(truth_weeks, pred_weeks), start=1):
            for lag in lags:
                a = _acf_one_week(truth, lag)
                b = _acf_one_week(pred, lag)
                error = abs(a - b)
                errors.append(error)
                records.append({'Model': name, 'Week_order_in_output': week, 'Lag_15min_steps': lag,
                                'ACF_truth': a, 'ACF_pred': b, 'ACF_abs_error': error})
        summary_rows.append({'Model': name, 'ACF_MAE': _finite_mean(errors),
                             'N_ACF_pairs_defined': int(np.isfinite(errors).sum()),
                             'N_ACF_pairs_undefined': int((~np.isfinite(errors)).sum())})
    return pd.DataFrame(summary_rows).set_index('Model'), pd.DataFrame(records)


def peak_rule_analysis(result_df):
    rows = []
    true_days = result_df['Truth'].to_numpy(dtype=float).reshape(-1, 96)
    for rule in ['first', 'last']:
        true_peak = np.argmax(true_days, axis=1) if rule == 'first' else 95 - np.argmax(true_days[:, ::-1], axis=1)
        for name in result_df.columns[1:]:
            pred_days = result_df[name].to_numpy(dtype=float).reshape(-1, 96)
            pred_peak = np.argmax(pred_days, axis=1) if rule == 'first' else 95 - np.argmax(pred_days[:, ::-1], axis=1)
            available = (true_days.max(axis=1) > 0) & (pred_days.max(axis=1) > 0)
            rows.append({'Model': name, 'Rule': rule,
                         'KS_including_zero_days': float(ks_2samp(true_peak, pred_peak).statistic),
                         'Peak_time_MAE_valid_min': _finite_mean(np.abs(pred_peak[available] - true_peak[available]) * 15),
                         'N_valid_days': int(available.sum()), 'N_excluded_days': int((~available).sum())})
    return pd.DataFrame(rows)


def additional_analysis_main():
    base = Path(__file__).resolve().parent
    folder = base / CONFIG['folder_path']
    configured = CONFIG['prediction_file']
    if configured:
        source = Path(configured)
        if not source.is_absolute():
            source = base / source
    else:
        matches = sorted(folder.glob('generate_data_*.xlsx'), key=lambda p: p.name)
        if not matches:
            raise FileNotFoundError(f'{folder} 没有generate_data_*.xlsx。请先运行main.py模式1。')
        source = matches[-1]  # 依据文件名时间戳，而非性能选择。
    print('只分析已有预测，不训练：', source)
    result_df = check_prediction_data(pd.read_excel(source, index_col=0))
    point = point_metrics_generate(result_df)
    daily, daily_pairs = daily_metrics_generate(result_df)
    acf, acf_pairs = acf_metrics_generate(result_df, CONFIG['acf_lags'])
    summary = pd.concat([point, daily, acf], axis=1)
    rules = peak_rule_analysis(result_df)
    output = base / CONFIG['outfolder'] / source.stem
    output.mkdir(parents=True, exist_ok=True)
    meta = {
        'source_file': str(source.resolve()),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'analysis_time': datetime.now().isoformat(timespec='seconds'),
        'analysis_version': 'V2_2_offline_metrics_from_V2_1',
        'n_weeks': len(result_df) // 672, 'n_days': len(result_df) // 96,
        'acf_lags': CONFIG['acf_lags'],
        'acf_definition': 'mean-centered autocovariance sum / lag0 sum, per week; undefined for constant weeks',
        'vf_zero_mean_rule': 'Same as original: daily VF=0 when daily mean is exactly zero; counts are reported',
        'peak_time_rule': 'last; paired timing excludes days without positive peaks in truth or prediction',
        'postprocessing': 'input is already postprocessed; NEVER run backward_process again',
        'training': 'not performed; this file cannot establish how source predictions were trained',
        'limitations': 'No raw prediction, branch ablation, training curve, or runtime can be reconstructed from final predictions alone',
    }
    # 沿用原工程pandas输出Excel的写法；不导入额外的训练/管理框架。
    with pd.ExcelWriter(output / 'additional_metrics.xlsx') as writer:
        summary.to_excel(writer, sheet_name='Summary')
        daily_pairs.to_excel(writer, sheet_name='Daily paired', index=False)
        acf_pairs.to_excel(writer, sheet_name='ACF by week', index=False)
        rules.to_excel(writer, sheet_name='Peak rules', index=False)
        pd.DataFrame({'Setting': list(meta), 'Value': [str(v) for v in meta.values()]}).to_excel(
            writer, sheet_name='Readme', index=False)
    summary.to_csv(output / 'additional_metrics.csv', encoding='utf-8-sig')
    daily_pairs.to_csv(output / 'daily_paired_metrics.csv', index=False, encoding='utf-8-sig')
    (output / 'analysis_source.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n附加指标（不改原主表）：')
    print(summary.round(4).to_string())
    print('\n保存位置：', output)
    print('只包含最终预测的离线分析；未生成训练日志、后处理前结果或分支消融。')
    return summary


if __name__ == '__main__':
    additional_analysis_main()
