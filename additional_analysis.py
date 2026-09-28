"""V2_2附加分析：保存后的预测、消融和后处理敏感性；不训练。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from additional_analysis_V21 import (check_prediction_data, point_metrics_generate,
    daily_metrics_generate, acf_metrics_generate, peak_rule_analysis)
from model_validation import compute_error_metrics, compute_daily_metrics, metrics_picture


def all_metrics(result_df):
    point = point_metrics_generate(result_df)
    daily, _ = daily_metrics_generate(result_df)
    acf, _ = acf_metrics_generate(result_df, [1, 2, 4, 8, 24, 48, 96])
    original = pd.concat([compute_error_metrics(result_df), metrics_picture(*compute_daily_metrics(result_df))])
    return pd.concat([original, pd.concat([point, daily, acf], axis=1).T])


# 原backward_process的诊断副本；主模型仍调用function_package里的原函数。
def postprocess_stage(result_df, round_set=True, clip_set=True):
    new_row = pd.DataFrame({col: [0] for col in result_df.columns})
    new_df = pd.concat([new_row, result_df], ignore_index=True)

    new_df['Diff'] = new_df['Generate'] - new_df['Truth']
    new_df['pred_new'] = new_df['Generate']
    new_df['pred_new'] = new_df['pred_new'].astype(float)

    total_length = len(new_df)
    for i in range(0, total_length, 4):
        diff_occ = new_df.loc[i, 'Diff']
        if 0 < i < total_length - 1:
            for j in range(-3, 4):
                new_df.loc[i + j, 'pred_new'] += - (1 - np.abs(j) / 4) * diff_occ
        if i == 0:
            for j in range(0, 4):
                new_df.loc[i + j, 'pred_new'] += - (1 - np.abs(j) / 4) * diff_occ
        if i == total_length - 1:
            for j in range(-3, 1):
                new_df.loc[i + j, 'pred_new'] += - (1 - np.abs(j) / 4) * diff_occ

    new_df = new_df.iloc[1:, :].copy()
    new_df.index = result_df.index
    if round_set:
        new_df['pred_new'] = new_df['pred_new'].round(0)
    if clip_set:
        new_df.loc[new_df['pred_new'] <= 0, 'pred_new'] = 0
    new_df = new_df[['pred_new']]

    return new_df


def postprocess_analysis(raw_df, final_df):
    if list(raw_df.columns) != list(final_df.columns) or not raw_df.index.equals(final_df.index):
        raise ValueError('后处理前后文件的列名或顺序不一致，不能混用不同轮结果。')
    if not np.array_equal(raw_df['Truth'].to_numpy(), final_df['Truth'].to_numpy()):
        raise ValueError('后处理前后真值不一致。')
    rows=[]
    for name in raw_df.columns[1:]:
        y=raw_df['Truth'].to_numpy(dtype=float)
        p=raw_df[name].to_numpy(dtype=float)
        stages={'Raw reconstruction':p}
        if name in ['Linear','Spline']:
            # 原插值函数内部已执行取整/截负；不能将合并时快照冒充完全原始输出。
            stages={'At merger (legacy rounding already applied)':p}
        else:
            pair=pd.DataFrame({'Truth':y,'Generate':p},index=raw_df.index)
            anchor=postprocess_stage(pair,False,False)['pred_new'].to_numpy()
            rounded=pd.Series(anchor).round(0).to_numpy()
            stages['Anchor only']=anchor
            stages['Anchor + rounding']=rounded
            check=np.where(rounded<=0,0,rounded)
            if not np.array_equal(check,final_df[name].to_numpy()):
                raise ValueError(name+'的最终结果不等于原后处理，需先检查源文件。')
        stages['Actual final']=final_df[name].to_numpy()
        for stage, values in stages.items():
            frame=pd.DataFrame({'Truth':y,name:values},index=raw_df.index)
            error=values-y; unknown=np.arange(len(y))%4!=3
            day_metrics=metrics_picture(*compute_daily_metrics(frame))
            rows.append(dict(Model=name, Stage=stage, RMSE=float(np.sqrt(np.mean(error**2))),
                MAE=float(np.mean(np.abs(error))), Unknown_RMSE=float(np.sqrt(np.mean(error[unknown]**2))),
                Negative_points=int(np.sum(values<0)),
                Noninteger_points=int(np.sum(np.abs(values-np.round(values))>1e-8)),
                **{k:day_metrics.loc[k,name] for k in day_metrics.index if k.endswith('-KS')}))
    return pd.DataFrame(rows)


def training_curve_picture(log_dir):
    import matplotlib.pyplot as plt
    folder=Path(log_dir); output=folder/'figures'; output.mkdir(exist_ok=True)
    for file in sorted(folder.glob('*_loss.csv')):
        history=pd.read_csv(file)
        fig,ax=plt.subplots(figsize=(6,4),dpi=300)
        ax.plot(history['epoch'],history['train_mse'],color='#C0362C',linewidth=1.4,label='Training MSE (dropout on)')
        ax.set_xlabel('Epoch');ax.set_ylabel('MSE')
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.grid(False);ax.legend(frameon=False);fig.tight_layout()
        fig.savefig(output/(file.stem+'.png'),dpi=300);plt.close(fig)


def additional_results_analysis(result_df, log_dir, picture_set=False, postprocess_set=True):
    result_df=check_prediction_data(result_df)
    folder=Path(log_dir)/'additional';folder.mkdir(parents=True,exist_ok=True)
    point=point_metrics_generate(result_df)
    daily,daily_pairs=daily_metrics_generate(result_df)
    acf,acf_pairs=acf_metrics_generate(result_df,[1,2,4,8,24,48,96])
    summary=pd.concat([point,daily,acf],axis=1)
    summary.to_csv(folder/'additional_metrics.csv',encoding='utf-8-sig')
    daily_pairs.to_csv(folder/'daily_paired_metrics.csv',index=False,encoding='utf-8-sig')
    with pd.ExcelWriter(folder/'additional_metrics.xlsx') as writer:
        summary.to_excel(writer,sheet_name='Summary')
        daily_pairs.to_excel(writer,sheet_name='Daily paired',index=False)
        acf_pairs.to_excel(writer,sheet_name='ACF by week',index=False)
        peak_rule_analysis(result_df).to_excel(writer,sheet_name='Peak rules',index=False)
    raw_file=Path(log_dir)/'predictions_before_postprocess.csv'
    if postprocess_set and raw_file.exists():
        raw=pd.read_csv(raw_file,index_col=0,float_precision='round_trip')
        raw.index=raw.index.astype(str);result_df.index=result_df.index.astype(str)
        sensitivity=postprocess_analysis(raw,result_df)
        sensitivity.to_csv(folder/'postprocess_sensitivity.csv',index=False,encoding='utf-8-sig')
        sensitivity.to_excel(folder/'postprocess_sensitivity.xlsx',index=False)
    notes={
      'version':'V2_2; no validation subset; train-MSE early stopping; last-epoch weights',
      'primary_metrics':'Original MSE-trained model, legacy LAST-maximum rule and VF/HF definitions unchanged.',
      'unknown':'Exclude zero-based 3::4 hourly anchors; 15/30/45-minute errors also reported.',
      'acf':'Per-week centered autocovariance / lag0 sum, lags 1,2,4,8,24,48,96. Constant-week pairs undefined and counted.',
      'peak_time':'Paired MAE uses days where BOTH truth/prediction have positive peaks. All-day KS retains original convention; first/last sensitivity included.',
      'vf':'Legacy rule: daily VF=0 if mean exactly 0; zero-day counts explicit. Raw negative outputs may yield physically invalid VF.',
      'scope':'Legacy interpolation, Markov, preprocessing, concatenated-week boundary conventions retained, not silently corrected.',
      'pvalues':'KS p-values are descriptive; p>0.05 does not establish equivalence; reused weeks are not independent experiments.',
      'postprocessing':'Final CSV is not postprocessed again. Linear/Spline snapshot is AFTER their original internal rounding.',
    }
    (folder/'metric_definitions.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    if picture_set: training_curve_picture(log_dir)
    print('附加指标已保存：',folder/'additional_metrics.xlsx')
    return summary


def paired_week_block_interval(result_df, week_indices, n_boot=1000, block_weeks=2, seed=2026):
    """固定已训练模型的测试时段抽样区间，不覆盖训练随机性/配置选择偏差。"""
    ids=np.asarray(week_indices,dtype=int)
    if len(ids)<2 or len(np.unique(ids))!=len(ids) or len(result_df)!=len(ids)*672:
        raise ValueError('需要至少两周、唯一周索引，并与预测顺序一一对应。')
    if block_weeks<1 or int(block_weeks)!=block_weeks or n_boot<2 or int(n_boot)!=n_boot:
        raise ValueError('block_weeks和n_boot必须是有效正整数。')
    if 'DWT-TimesNet' not in result_df: raise ValueError('缺少DWT-TimesNet列。')
    order=np.argsort(ids);sid=ids[order]
    segments=np.split(np.arange(len(ids)),np.flatnonzero(np.diff(sid)!=1)+1)
    blocks=[p[i:i+block_weeks] for p in segments for i in range(0,len(p),block_weeks)]
    unknown=np.arange(672)%4!=3
    truth=result_df['Truth'].to_numpy().reshape(-1,672)[order]
    stats={}
    _,peaks,vfs,_=compute_daily_metrics(result_df)
    true_p=peaks['Truth'].to_numpy().reshape(-1,7)[order]
    true_v=vfs['Truth'].to_numpy().reshape(-1,7)[order]
    for name in result_df.columns[1:]:
        error=result_df[name].to_numpy().reshape(-1,672)[order]-truth
        stats[name]=(np.abs(error[:,unknown]).mean(axis=1),(error[:,unknown]**2).mean(axis=1),
                     peaks[name].to_numpy().reshape(-1,7)[order],vfs[name].to_numpy().reshape(-1,7)[order])
    def measures(name,ix):
        mae,mse,peak,vf=stats[name]
        return np.array([mae[ix].mean(),np.sqrt(mse[ix].mean()),np.abs(peak[ix]-true_p[ix]).mean(),
              np.abs(vf[ix]-true_v[ix]).mean(),ks_2samp(true_p[ix].ravel(),peak[ix].ravel()).statistic,
              ks_2samp(true_v[ix].ravel(),vf[ix].ravel()).statistic])
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(int(n_boot)):
        selected=[]
        while len(selected)<len(ids): selected.extend(blocks[int(rng.integers(len(blocks)))].tolist())
        samples.append(np.asarray(selected[:len(ids)],dtype=int))
    names=['Unknown-MAE','Unknown-RMSE','Daily-peak-MAE','Daily-VF-MAE','Peak-KS','VF-KS'];rows=[]
    dwt_samples=np.array([measures('DWT-TimesNet',ix) for ix in samples])
    original=np.arange(len(ids))
    for name in stats:
        if name=='DWT-TimesNet':continue
        changes=dwt_samples-np.array([measures(name,ix) for ix in samples])
        effect=measures('DWT-TimesNet',original)-measures(name,original)
        for j,metric in enumerate(names):
            rows.append({'Baseline':name,'Metric':metric,'DWT minus baseline':effect[j],
              'CI2.5%':np.quantile(changes[:,j],.025),'CI97.5%':np.quantile(changes[:,j],.975),
              'Blocks':len(blocks),'Singleton blocks':sum(len(b)==1 for b in blocks),
              'Block weeks requested':block_weeks,'Bootstrap replicates':n_boot})
    return pd.DataFrame(rows)


def analyse_saved_run(log_dir, bootstrap=False, n_boot=1000, block_weeks=2):
    folder=Path(log_dir)
    result=pd.read_csv(folder/'predictions.csv',index_col=0)
    additional_results_analysis(result,folder,picture_set=True)
    from runtime_analysis import runtime_summary
    runtime_summary(folder)
    if bootstrap:
        settings=json.loads((folder/'run_settings.json').read_text(encoding='utf-8'))
        ci=paired_week_block_interval(result,settings['test'],n_boot,block_weeks,int(settings['seed']))
        ci.to_excel(folder/'additional'/'paired_block_intervals.xlsx',index=False)
    return result


if __name__=='__main__':
    folder=input('请输入training_runs下的案例目录（不会训练）：').strip().strip('"')
    analyse_saved_run(folder)
