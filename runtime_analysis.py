"""V2_2：被动记录训练、随机状态、运行时间；不负责训练或选择模型。"""
import ast
import csv
import hashlib
import json
import os
import platform
import random
import time
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert), encoding='utf-8')


def time_start(device=None):
    if device is not None and torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def time_end(start, device=None):
    if device is not None and torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def rng_snapshot():
    np_state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
            'torch': torch.get_rng_state().tolist(),
            'cuda': [s.tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng_state(state):
    def tuple_tree(value):
        return tuple(tuple_tree(v) for v in value) if isinstance(value, list) else value
    random.setstate(tuple_tree(state['python']))
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), int(n[2]), int(n[3]), float(n[4])))
    torch.set_rng_state(torch.tensor(state['torch'], dtype=torch.uint8))
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('保存的CUDA随机状态与当前设备数量不一致；请在同一环境复现。')
        torch.cuda.set_rng_state_all([torch.tensor(v, dtype=torch.uint8) for v in state['cuda']])


@contextmanager
def isolated_rng(seed=None, state_file=None):
    """新增模型/消融不消耗主流程随机状态；不改变原五种方法的初始化。"""
    state = rng_snapshot()
    try:
        if state_file is not None:
            restore_rng_state(json.loads(Path(state_file).read_text(encoding='utf-8')))
        elif seed is not None:
            random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        restore_rng_state(state)


def save_rng_state(log_dir, name):
    if log_dir is not None:
        write_json(Path(log_dir) / ('rng_' + name + '.json'), rng_snapshot())


def _rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except (ImportError, OSError):
        return None


def start_training_record(model, device):
    if torch.device(device).type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    model.training_start_time = time_start(device)


def finish_training_record(model, device):
    model.training_seconds = time_end(model.training_start_time, device)
    model.cuda_peak_process_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                                 if torch.device(device).type == 'cuda' else None)


def save_training_record(model, history, log_dir, name, model_params, train_params,
                         train_test_index, save_weights=False):
    model.training_history = history
    model.selected_epoch = len(history)
    model.split_indices = {'train': np.asarray(train_test_index[0]).tolist(),
                           'test': np.asarray(train_test_index[1]).tolist()}
    if log_dir is None: return
    folder = Path(log_dir); folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(folder / (name + '_loss.csv'), index=False, encoding='utf-8-sig')
    values = [r['train_mse'] for r in history]
    record = {'model': name, 'loss': 'MSELoss', 'early_stopping_monitor': 'training_mse',
              'validation_weeks': 0, 'restore_best_weights': False,
              'selected_epoch': len(history), 'epochs_run': len(history),
              'minimum_recorded_training_epoch': int(np.argmin(values)) + 1 if values else None,
              'minimum_recorded_training_mse': min(values) if values else None,
              'last_recorded_training_mse': values[-1] if values else None,
              'loss_history_meaning': 'Dropout-on, pre-update loss; not re-evaluated MSE of the saved final weights.',
              'timesnet_epoch_mean': 'Unweighted mean of batch MSE, as in V2_1; default batch_size=1.',
              'training_seconds': model.training_seconds, 'model_params': model_params,
              'train_params': train_params, 'split_indices': model.split_indices,
              'test_used_for_stopping': False,
              'stop_reason': 'patience' if len(history) < train_params['num_epochs'] else 'epoch_limit'}
    write_json(folder / (name + '_training.json'), record)
    if save_weights:
        # 只保存末轮；绝不load_state_dict或恢复最小损失轮。
        torch.save(model.state_dict(), folder / (name + '_last_weights.pt'))
    print(f'{name}: kept last epoch {len(history)}; early stopping uses training MSE; validation=0.')


def _runtime_row(log_dir, row):
    if log_dir is None: return
    path = Path(log_dir) / 'model_runtime.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['model', 'group', 'trainable_parameters', 'parameter_memory_mb', 'training_seconds',
              'inference_seconds', 'test_evaluation_seconds', 'epochs_run', 'selected_epoch',
              'evaluation_weeks', 'inference_ms_per_week', 'device', 'rss_after_model_mb', 'cuda_peak_process_mb']
    old = []
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as stream:
            old = [r for r in csv.DictReader(stream) if r['model'] != row['model']]
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(old + [row])


def save_model_runtime(model, log_dir, name, group, evaluation_weeks):
    inference = getattr(model, 'inference_seconds', None)
    _runtime_row(log_dir, {'model': name, 'group': group,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'parameter_memory_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2,
        'training_seconds': getattr(model, 'training_seconds', None),
        'inference_seconds': inference, 'test_evaluation_seconds': getattr(model, 'test_evaluation_seconds', 0.0),
        'epochs_run': len(getattr(model, 'training_history', [])), 'selected_epoch': getattr(model, 'selected_epoch', None),
        'evaluation_weeks': evaluation_weeks,
        'inference_ms_per_week': inference * 1000 / evaluation_weeks if inference is not None and evaluation_weeks else None,
        'device': str(next(model.parameters()).device), 'rss_after_model_mb': _rss_mb(),
        'cuda_peak_process_mb': getattr(model, 'cuda_peak_process_mb', None)})


def save_baseline_runtime(log_dir, name, training_seconds, inference_seconds, n_weeks):
    _runtime_row(log_dir, {'model': name, 'group': name, 'trainable_parameters': 0,
        'parameter_memory_mb': None, 'training_seconds': training_seconds, 'inference_seconds': inference_seconds,
        'test_evaluation_seconds': 0, 'epochs_run': 0, 'selected_epoch': None, 'evaluation_weeks': n_weeks,
        'inference_ms_per_week': inference_seconds * 1000 / n_weeks if n_weeks else None,
        'device': 'cpu', 'rss_after_model_mb': _rss_mb(), 'cuda_peak_process_mb': None})


def save_stage_runtime(log_dir, stage, seconds):
    if log_dir is None: return
    path = Path(log_dir) / 'stage_runtime.csv'; path.parent.mkdir(parents=True, exist_ok=True)
    first = not path.exists()
    with path.open('a', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        if first: writer.writerow(['stage', 'wall_seconds'])
        writer.writerow([stage, seconds])


def runtime_summary(log_dir):
    folder = Path(log_dir); source = folder / 'model_runtime.csv'
    if not source.exists(): return
    detail = pd.read_csv(source)
    rows = []
    for group, d in detail.groupby('group', sort=False):
        n = int(d['evaluation_weeks'].iloc[0]); infer = d['inference_seconds'].sum(min_count=1)
        rows.append({'Model': group, 'Trainable parameters': d['trainable_parameters'].sum(min_count=1),
                     'Parameter memory MB': d['parameter_memory_mb'].sum(min_count=1),
                     'Training seconds': d['training_seconds'].sum(min_count=1), 'Inference seconds': infer,
                     'Inference ms per week': infer * 1000 / n if n else None,
                     'Evaluation weeks': n, 'Device': ', '.join(d['device'].astype(str).unique())})
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / 'runtime_comparison.csv', index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(folder / 'runtime_comparison.xlsx') as writer:
        summary.to_excel(writer, sheet_name='Models', index=False)
        detail.to_excel(writer, sheet_name='Branches', index=False)
        stage = folder / 'stage_runtime.csv'
        if stage.exists(): pd.read_csv(stage).to_excel(writer, sheet_name='Stage wall times', index=False)


def source_fingerprint():
    root = Path(__file__).resolve().parent
    names = ['CNN_Process.py', 'DWT_CNN_TimesNet.py', 'TimesNet.py', 'TimesNet_Process.py',
             'function_package.py', 'model_package.py', 'Markov_Process.py',
             'test_layers/Embed.py', 'test_layers/Conv_Blocks.py', 'model_validation.py']
    hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}
    main_tree = ast.parse((root / 'function_main.py').read_text(encoding='utf-8-sig'))
    body = [n for n in main_tree.body if isinstance(n, ast.FunctionDef)]
    hashes['function_main_functions'] = hashlib.sha256(
        ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    return hashes


def run_identity(config, high_list, split, seed):
    # 不将输出路径、续跑区间和图片开关算成模型超参数；其余设置全部保留。
    excluded = {'outfile', 'picture_set', 'seed_start', 'seed_stop', 'skip_completed',
                'additional_set', 'save_weights', 'ablation_set'}
    cfg = {k:v for k,v in config.items() if k not in excluded}
    h = hashlib.sha256()
    for frame in high_list:
        h.update(frame['Occupancy'].to_numpy(dtype=np.float64).tobytes())
        h.update('\n'.join(map(str,frame.index)).encode('utf-8'))
    info = {'version':'V2_2', 'seed': int(seed), 'config':cfg, 'data_sha256':h.hexdigest(),
            'train': np.asarray(split[0]).tolist(), 'test': np.asarray(split[1]).tolist(),
            'source':source_fingerprint()}
    identity = hashlib.sha256(json.dumps(info, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return identity, info


def create_run(config, high_list, split, seed, tag='one'):
    train, test = [np.asarray(x, dtype=int) for x in split]
    if not len(train) or not len(test) or len(set(train) & set(test)):
        raise ValueError('训练/测试必须非空、不重叠。')
    if len(set(train) | set(test)) != len(high_list):
        raise ValueError('训练/测试未覆盖全部周样本。')
    identity, info = run_identity(config, high_list, split, seed)
    root = Path(config['outfile']); runs = root / 'training_runs'; runs.mkdir(parents=True, exist_ok=True)
    if config.get('skip_completed', False):
        for old in sorted(runs.glob(tag + '_seed' + str(seed) + '_*'), reverse=True):
            complete = old / 'completed.json'
            if complete.exists():
                saved = json.loads(complete.read_text(encoding='utf-8'))
                files = saved.get('files', {})
                if saved.get('identity') == identity and files and all(
                    (old / n).is_file() and hashlib.sha256((old/n).read_bytes()).hexdigest() == digest
                    for n, digest in files.items()):
                    print('跳过相同配置下已完成的轮次：', old)
                    return None
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder = runs / f'{tag}_seed{seed}_{stamp}'; folder.mkdir()
    settings = dict(info, identity=identity, n_train=len(train), n_validation=0, n_test=len(test),
                    early_stop_monitor='training_mse', weight_selection='last_completed_epoch',
                    output_config=config, split_kind='chronological' if tag=='time' else tag)
    write_json(folder / 'run_settings.json', settings)
    rows=[]
    for role, ids in [('train',train),('test',test)]:
        for order,i in enumerate(ids):
            rows.append({'role':role,'order_in_role':order,'week_index_0based':int(i),
                         'first_label':str(high_list[i].index[0]),'last_label':str(high_list[i].index[-1])})
    pd.DataFrame(rows).to_csv(folder/'split_weeks.csv',index=False,encoding='utf-8-sig')
    versions={}
    for n in ['numpy','pandas','scipy','torch','PyWavelets','matplotlib','openpyxl','tqdm']:
        try: versions[n]=metadata.version(n)
        except metadata.PackageNotFoundError: versions[n]='not found'
    write_json(folder/'environment.json', {'python':platform.python_version(),'platform':platform.platform(),
       'processor':platform.processor() or 'not reported','logical_cpu_count':os.cpu_count(),
       'torch_threads':torch.get_num_threads(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       'versions':versions,
       'timing':'Training loop only (train-MSE monitoring, no validation, no best-weight restore). Inference excludes plotting. Stage times may include data/plots/IO. No warmup or repeated benchmark.',
       'memory':'Parameter storage only; RSS is process snapshot, not peak; CUDA peak is total process allocation, not incremental per-model footprint. Missing is not zero.'})
    print(f'V2_2 split: train={len(train)}, validation=0, test={len(test)}; training MSE / last epoch.')
    print('Training log directory:', folder)
    return folder


def mark_completed(log_dir):
    folder=Path(log_dir); meta=json.loads((folder/'run_settings.json').read_text(encoding='utf-8'))
    required=['predictions.csv','main_metrics.csv','predictions_before_postprocess.csv','runtime_comparison.csv']
    optional=['DWT_components.csv','additional/additional_metrics.csv']
    files={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in required if (folder/n).is_file()}
    if len(files)!=len(required): raise RuntimeError('主结果存档不完整，不能标为完成。')
    for n in optional:
        if (folder/n).exists(): files[n]=hashlib.sha256((folder/n).read_bytes()).hexdigest()
    refs = folder / 'result_files.json'
    if refs.exists():
        files['result_files.json'] = hashlib.sha256(refs.read_bytes()).hexdigest()
        main_file = Path(json.loads(refs.read_text(encoding='utf-8'))['main_result'])
        if main_file.exists():
            relative = os.path.relpath(main_file, folder)
            files[relative] = hashlib.sha256(main_file.read_bytes()).hexdigest()
    write_json(folder/'completed.json',{'identity':meta['identity'],'files':files})
