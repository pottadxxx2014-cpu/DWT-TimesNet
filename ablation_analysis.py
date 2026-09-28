"""V2_2独立消融：默认复用本版本分支；可按同一初始化重训保留分支。"""
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

BRANCHES = {
    'Full': ('Low', 'D1', 'D2'),
    'Low-only': ('Low',),
    'Without-D1': ('Low', 'D2'),
    'Without-D2': ('Low', 'D1'),
    'Without-CNN-zero-low': ('D1', 'D2'),
    'Fixed-low-plus-high': ('D1', 'D2'),
}


def resolve_source_run(config, require_components=True):
    root = Path(__file__).resolve().parent
    if config.get('source_run'):
        p = Path(config['source_run'])
        candidates = [p if p.is_absolute() else root / p]
    else:
        folder = Path(config['source_folder'])
        if not folder.is_absolute(): folder = root / folder
        candidates = sorted((folder / 'training_runs').glob('*_seed' + str(config['seed']) + '_*'), reverse=True)
    for path in candidates:
        settings = path / 'run_settings.json'
        if not settings.exists(): continue
        meta = json.loads(settings.read_text(encoding='utf-8'))
        required = path / ('DWT_components.csv' if require_components else 'predictions.csv')
        if meta.get('version') == 'V2_2' and int(meta.get('seed',-1)) == int(config['seed']) and required.exists():
            if meta.get('n_validation') != 0 or meta.get('early_stop_monitor') != 'training_mse':
                raise ValueError('来源不是本版训练MSE/无验证集协议。')
            print('读取来源（按时间，不按性能选择）：', path)
            return path
    raise FileNotFoundError('没有找到对应seed的V2_2结果。先运行本版模式1；旧V2_1/V3不能自动混入。')


def fixed_low_from_anchors(truth, wavelet, level):
    """只读取3::4整点；未知15min真值绝不进入固定低频构造。"""
    from function_package import interpolate_occupancy_zero, DWT_week
    values = np.asarray(truth,dtype=float)
    if len(values) == 0 or len(values) % 672: raise ValueError('固定低频需要完整周。')
    fixed = []
    for week in values.reshape(-1,672):
        hourly = pd.DataFrame({'Occupancy':week[3::4]})
        expanded = interpolate_occupancy_zero(hourly)
        _, comp = DWT_week(pd.DataFrame({'Occupancy':expanded}), wavelet=wavelet, level=level+2, picture=False)
        fixed.append((comp['Occupancy']-comp['D1']-comp['D2']).to_numpy())
    return np.concatenate(fixed)


def compose_ablation(components, fixed_low, variants):
    if not variants or len(set(variants)) != len(variants) or any(n not in BRANCHES for n in variants):
        raise ValueError('variant_list必须是无重复的已知消融名称。')
    n = len(components['Truth'])
    if n == 0 or n % 672: raise ValueError('消融需要非空完整周。')
    arrays = {k:np.asarray(components[k],dtype=(float if k=='Truth' else np.float32))
              for k in ['Truth','Low','D1','D2']}
    if any(v.shape != (n,) or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('分支长度或数值非法。')
    fixed = np.asarray(fixed_low,dtype=float)
    if fixed.shape != (n,): raise ValueError('固定低频长度不一致。')
    result={'Truth':arrays['Truth']}
    for name in variants:
        parts=BRANCHES[name]
        if name == 'Fixed-low-plus-high':
            pred=fixed.copy()
            for k in parts: pred=pred+arrays[k]
        else:
            pred=arrays[parts[0]].copy()
            for k in parts[1:]: pred=pred+arrays[k]
        result[name] = pred
    return pd.DataFrame(result,index=components.index)


def apply_postprocessing(raw):
    from function_package import backward_process
    final = raw.copy()
    for name in raw.columns[1:]:
        pair = pd.DataFrame({'Truth':raw['Truth'],'Generate':raw[name]},index=raw.index)
        final[name] = backward_process(pair)['pred_new'].to_numpy()
    return final


def _read_training_source(source, meta):
    from function_package import read_data, DWT_process, DWT_two_frequency_method_process, index_generating
    from runtime_analysis import run_identity
    cfg = meta['output_config']
    if meta['split_kind'] not in ['one','cross','time']:
        raise ValueError('重训消融目前只支持主办公室Raw/Month；跨楼层/外部案例可复用已有分支。')
    root=Path(__file__).resolve().parent
    file=root/('occupancy_office_month.xlsx' if cfg['file_set']=='Month' else 'occupancy_office_rawdata.xlsx')
    high,low,n=read_data(file,ignore_cols=3)
    split=(np.asarray(meta['train'],dtype=int),np.asarray(meta['test'],dtype=int))
    identity,_=run_identity(cfg,high,split,meta['seed'])
    if identity != meta['identity']:
        raise ValueError('当前模型源码/数据/配置与来源不一致，不能把重训结果作为受控消融。请使用相同版本的源案例。')
    labels=index_generating(high,split)
    counts,_,low_comp,high_comp=DWT_process(low,high,n,cfg['wavelet_set'],cfg['level_set'],picture=False)
    x1,x2,y1,y2=DWT_two_frequency_method_process(counts,low_comp,high_comp,n,picture=False)
    return low_comp,high_comp,n,split,labels,x1,x2,y1,y2


def retrain_variant(source, meta, name, data, components, fixed_low, folder):
    import torch
    from runtime_analysis import isolated_rng, time_start, time_end, save_model_runtime, runtime_summary
    from DWT_CNN_TimesNet import CNN_based_transform, TimesNet_for_DWT, timesnet_pred_generate
    low,high,n,split,labels,x1,x2,y1,y2=data
    enabled=BRANCHES[name]
    output={k:np.zeros(len(components)) for k in ['Low','D1','D2']}
    for branch in enabled:
        key={'Low':'DWT_CNN','D1':'TimesNet_Max','D2':'TimesNet_Middle'}[branch]
        record=json.loads((source/(key+'_training.json')).read_text(encoding='utf-8'))
        rng_file=source/('rng_'+key+'.json')
        if not rng_file.exists(): raise FileNotFoundError('缺少原分支初始化随机状态：'+str(rng_file))
        with isolated_rng(state_file=rng_file):
            if branch=='Low':
                model,test_data,_=CNN_based_transform(low,high,n,split,record['model_params'],record['train_params'],
                                                       'Zero',False,log_dir=folder)
                device=next(model.parameters()).device
                start=time_start(device)
                model.eval()
                with torch.no_grad(): pred=model(test_data[0].to(device))
                output[branch]=pred.cpu().numpy()[:,0,:,:].transpose(0,2,1).reshape(-1)
                model.inference_seconds=time_end(start,device)
                save_model_runtime(model,folder,'DWT_CNN','DWT-TimesNet',len(split[1]))
            else:
                branch_name='Max' if branch=='D1' else 'Middle'
                x,y=(x1,y1) if branch=='D1' else (x2,y2)
                model,loader=TimesNet_for_DWT(x,y,split,branch_name,record['model_params'],record['train_params'],log_dir=folder)
                pred,_,_=timesnet_pred_generate(loader,model,branch_name,'Blue',next(model.parameters()).device,False,log_dir=folder)
                output[branch]=pred
        del model
    if name=='Fixed-low-plus-high':
        reconstructed=fixed_low.copy()
        for k in enabled: reconstructed=reconstructed+output[k]
    else:
        reconstructed=output[enabled[0]].copy()
        for k in enabled[1:]: reconstructed=reconstructed+output[k]
    pd.DataFrame(output,index=components.index).to_csv(folder/'retrained_components.csv',encoding='utf-8-sig')
    runtime_summary(folder)
    return reconstructed


def ablation_main(config):
    from additional_analysis import all_metrics
    from runtime_analysis import write_json
    source=resolve_source_run(config)
    meta=json.loads((source/'run_settings.json').read_text(encoding='utf-8'))
    comp=pd.read_csv(source/'DWT_components.csv',index_col=0,float_precision='round_trip')
    if len(comp) != len(meta['test'])*672: raise ValueError('分支文件长度与源测试周数不一致。')
    required=['Truth','Low','D1','D2','Low_truth','D1_truth','D2_truth']
    if any(k not in comp for k in required): raise ValueError('分支文件列不完整。')
    if not np.allclose(comp['Low_truth']+comp['D1_truth']+comp['D2_truth'],comp['Truth'],atol=1e-5,rtol=1e-6):
        raise ValueError('真值分量之和与原真值不一致，请检查版本或索引。')
    cfg=meta['output_config']
    fixed=fixed_low_from_anchors(comp['Truth'].to_numpy(),cfg['wavelet_set'],cfg['level_set'])
    variants=config.get('variant_list',list(BRANCHES))
    raw=compose_ablation(comp,fixed,variants)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    strategy='retrain' if config.get('retrain',False) else 'reuse'
    folder=source/'ablation'/f'{strategy}_{stamp}';folder.mkdir(parents=True)
    if strategy=='retrain':
        data=_read_training_source(source,meta)
        for name in variants:
            print('只重训该变体启用的分支：',name,BRANCHES[name])
            child=folder/name;child.mkdir()
            raw[name]=retrain_variant(source,meta,name,data,comp,fixed,child)
    final=apply_postprocessing(raw)
    raw.to_csv(folder/'ablation_predictions_raw.csv',encoding='utf-8-sig')
    final.to_csv(folder/'ablation_predictions_final.csv',encoding='utf-8-sig')
    pd.DataFrame({'Fixed_low':fixed},index=comp.index).to_csv(folder/'fixed_low_from_hourly.csv',encoding='utf-8-sig')
    # 预测优先落盘，即使Excel被占用也不必再训练。
    before,after=all_metrics(raw),all_metrics(final)
    before.to_csv(folder/'ablation_metrics_before.csv',encoding='utf-8-sig')
    after.to_csv(folder/'ablation_metrics_after.csv',encoding='utf-8-sig')
    with pd.ExcelWriter(folder/'ablation_results.xlsx') as writer:
        before.to_excel(writer,sheet_name='Before postprocessing')
        after.to_excel(writer,sheet_name='After postprocessing')
    check=None
    predfile=source/'predictions.csv'
    if 'Full' in final and predfile.exists():
        old=pd.read_csv(predfile,index_col=0)
        if old.index.astype(str).tolist()!=final.index.astype(str).tolist(): raise ValueError('来源最终预测与分支索引不一致。')
        check=float(np.max(np.abs(old['DWT-TimesNet'].to_numpy()-final['Full'].to_numpy())))
        if strategy=='reuse' and check!=0: raise ValueError('复用Full未重现来源最终预测；不能合并报告。')
    write_json(folder/'ablation_protocol.json',{
        'version':'V2_2','source_run':str(source.resolve()),'source_identity':meta['identity'],
        'strategy':strategy,'variants':{n:list(BRANCHES[n]) for n in variants},
        'validation_weeks':0,'early_stop_monitor':'training_mse','weights':'last_completed_epoch',
        'training_targets':'Original per-component targets retained; no remaining branch is re-targeted to learn the removed component.',
        'rng':'reuse: exact stored predictions. retrain: restore source pre-initialization RNG per enabled branch; no model weights restored.',
        'postprocess':'Reconstruct each raw variant FIRST, then call original backward_process once per variant.',
        'fixed_low':'Hourly observations only: zero-order expansion then DWT, remove high-resolution D1,D2.',
        'Full_final_max_absolute_difference_from_source':check,
        'interpretation':'Independent-branch deletion/replacement, not joint retraining or capacity matching. Hardware numerical differences may affect retraining.'})
    print('\n消融完成：',folder/'ablation_results.xlsx')
    print('训练策略：',strategy,'；每个变体已单独执行原后处理。')
    return final
