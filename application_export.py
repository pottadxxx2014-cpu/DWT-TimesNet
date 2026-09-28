"""只导出已有预测给下游仿真，不生成假负荷，也不训练模型。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def export_application_inputs(run_dir):
    folder=Path(run_dir)
    settings=json.loads((folder/'run_settings.json').read_text(encoding='utf-8'))
    result=pd.read_csv(folder/'predictions.csv',index_col=0)
    if settings.get('split_kind') not in ['one','cross','time']:
        raise ValueError('此导出仅支持主办公室周索引，外部楼层数据需额外提供楼层/日期映射。')
    ids=np.asarray(settings['test'],dtype=int)
    if len(result)!=672*len(ids):raise ValueError('周索引与预测长度不匹配。')
    order=np.argsort(ids); ordered=ids[order]
    segments=np.split(order,np.flatnonzero(np.diff(ordered)!=1)+1)
    output=folder/'application_inputs';output.mkdir(exist_ok=True)
    records=[]
    for j,segment in enumerate(segments,1):
        rows=[]
        for position in segment:
            frame=result.iloc[int(position)*672:(int(position)+1)*672].copy()
            frame.insert(0,'Source_label',frame.index.astype(str))
            frame.insert(0,'Week_index_0based',int(ids[position]))
            rows.append(frame)
        data=pd.concat(rows,ignore_index=True)
        data.insert(0,'Step_15min',np.arange(len(data)))
        name=f'continuous_segment_{j:03d}.csv'
        data.to_csv(output/name,index=False,encoding='utf-8-sig')
        records.append({'file':name,'first_week':int(ids[segment[0]]),'last_week':int(ids[segment[-1]]),'n_weeks':len(segment)})
    pd.DataFrame(records).to_csv(output/'segments.csv',index=False,encoding='utf-8-sig')
    (output/'README.txt').write_text(
        '各CSV内部源周连续，缺周处拆段；不能直接拼成年份。Source_label可能无年份，未虚构绝对日期。\n'
        '仿真前需核对真实日期/天气及预热、初始状态，所有方法使用相同建筑/控制设置。\n'
        '本模块只导出人数预测；未执行建筑、电力、储能或需求响应模型，不能作为应用验证结果。\n',encoding='utf-8')
    print('应用输入导出：',output)
    return output


if __name__=='__main__':
    export_application_inputs(input('请输入training_runs下的完整案例目录：').strip().strip('"'))
