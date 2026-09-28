"""Serializable competition-facing aggregate reports and CSV exports."""
from __future__ import annotations
import csv, json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from .evaluator import EvaluationRecord

def _size_bucket(num_tasks:int)->str:
    return 'small' if num_tasks<=50 else 'medium' if num_tasks<=100 else 'large'

def _group_rows(values:list[EvaluationRecord], field:str)->list[dict[str,Any]]:
    groups:dict[str,list[float]]={}
    for item in values:
        key=_size_bucket(item.num_tasks) if field=='size' else str(getattr(item,field))
        groups.setdefault(key,[]).append(item.ratio)
    return [{field:key,'count':len(ratios),'mean_ratio':float(np.mean(ratios)),'std_ratio':float(np.std(ratios)),'median_ratio':float(np.median(ratios))} for key,ratios in sorted(groups.items())]

def summarize(records: Iterable[EvaluationRecord], *, model:str, split:str, seed:int, config_hash:str, training_time_seconds:float=0.0, bootstrap_samples:int=0)->dict[str,Any]:
    values=list(records)
    if not values: raise ValueError('cannot summarize an empty evaluation')
    ratios=np.asarray([x.ratio for x in values],dtype=float); makespans=np.asarray([x.makespan for x in values],dtype=float); heft=np.asarray([x.heft_makespan for x in values],dtype=float)
    datasets=sorted({x.dataset_source for x in values})
    comparison_tolerance=1e-9
    summary={'model':model,'split':split,'num_scenarios':len(values),'num_base_dags':len({x.base_dag_id for x in values}),'datasets':datasets,'mean_makespan':float(makespans.mean()),'std_makespan':float(makespans.std()),'mean_heft_makespan':float(heft.mean()),'mean_ratio':float(ratios.mean()),'std_ratio':float(ratios.std()),'median_ratio':float(np.median(ratios)),'p90_ratio':float(np.quantile(ratios,.9)),'min_ratio':float(ratios.min()),'max_ratio':float(ratios.max()),'better_than_heft_rate':float(np.mean(ratios<1.0-comparison_tolerance)),'equal_to_heft_rate':float(np.mean(np.abs(ratios-1.0)<=comparison_tolerance)),'valid_schedule_rate':float(np.mean([x.valid_schedule for x in values])),'mean_inference_time_ms':float(np.mean([x.inference_time_ms for x in values])),'training_time_seconds':float(training_time_seconds),'seed':seed,'config_hash':config_hash}
    if bootstrap_samples>0:
        rng=np.random.default_rng(seed); samples=np.asarray([rng.choice(ratios,len(ratios),replace=True).mean() for _ in range(bootstrap_samples)])
        summary['mean_ratio_ci95']=[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]
    return summary

def write_report(output_dir:str|Path, records:Iterable[EvaluationRecord], summary:dict[str,Any])->None:
    target=Path(output_dir); target.mkdir(parents=True,exist_ok=True); values=list(records)
    with (target/'per_scene.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(asdict(values[0]))); writer.writeheader(); writer.writerows(asdict(x) for x in values)
    (target/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    with (target/'summary.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    rows=[]
    for source in sorted({x.dataset_source for x in values}):
        subset=[x.ratio for x in values if x.dataset_source==source]; rows.append({'dataset_source':source,'count':len(subset),'mean_ratio':float(np.mean(subset)),'std_ratio':float(np.std(subset))})
    with (target/'ratio_by_dataset.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['dataset_source','count','mean_ratio','std_ratio']); writer.writeheader(); writer.writerows(rows)
    for filename,field in (('ratio_by_size.csv','size'),('ratio_by_domain.csv','domain')):
        group_rows=_group_rows(values,field)
        with (target/filename).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(group_rows[0])); writer.writeheader(); writer.writerows(group_rows)
    _write_plots(target, values)

def _write_plots(target:Path, values:list[EvaluationRecord])->None:
    """Best-effort plots; CSV/JSON reporting never depends on Matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ratios=[item.ratio for item in values]
    lower, upper = min(ratios), max(ratios)
    scale = max(1.0, abs(lower), abs(upper))
    histogram_range = (lower - 0.01 * scale, upper + 0.01 * scale) if upper - lower <= 1e-9 * scale else None
    figure, axis = plt.subplots(figsize=(6, 4))
    try:
        axis.hist(ratios, bins=min(20, max(1, len(ratios))), range=histogram_range)
        axis.set(xlabel='Policy / HEFT makespan ratio', ylabel='count', title='Ratio histogram')
        figure.tight_layout()
        figure.savefig(target / 'ratio_histogram.png', dpi=150)
    finally:
        plt.close(figure)

def write_training_curve(output_dir:str|Path, history:Iterable[dict[str,Any]])->None:
    """Write a validation-ratio curve when training emitted evaluation rows."""
    rows=list(history)
    if not rows: return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    points=[]
    for row in rows:
        ratio=row.get('proxy_validation_mean_ratio',row.get('validation_mean_ratio'))
        if ratio is None or not np.isfinite(ratio):
            ratio=row.get('validation_mean_ratio')
        # Training history rows carry `global_step`; a selected-best checkpoint
        # row may only carry `step`.  Either identifies the evaluation point.
        step=row.get('step',row.get('global_step'))
        if ratio is not None and np.isfinite(ratio) and step is not None: points.append(dict(step=step,ratio=ratio))
    if not points: return
    figure,axis=plt.subplots(figsize=(6,4)); axis.plot([row['step'] for row in points],[row['ratio'] for row in points],marker='o',label='Selection evaluation'); axis.plot([row['step'] for row in points],np.minimum.accumulate([row['ratio'] for row in points]),linestyle='--',label='Best so far'); axis.set(xlabel='training step',ylabel='mean ratio',title='Selection history (full validation or configured proxy)'); axis.legend(); figure.tight_layout(); figure.savefig(Path(output_dir)/'training_curve.png',dpi=150); plt.close(figure)
