"""Incremental, flushed experiment events and atomic machine-readable status."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class ProgressLog:
    def __init__(self, output, method, seed):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.method = method
        self.seed = seed
        self.started = perf_counter()
        self.state = dict(status='running', method=method, seed=seed, test_evaluated=False)
        self.handle = (self.output / 'events.jsonl').open('x', encoding='utf-8')

    def __enter__(self):
        self.emit('starting')
        return self

    def emit(self, event, **values):
        record = dict(time=datetime.now(timezone.utc).isoformat(), method=self.method,
                      seed=self.seed, event=event, elapsed_seconds=round(perf_counter() - self.started, 3), **values)
        self.handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.handle.flush()
        self.state.update(record)
        atomic_json(self.output / 'status.json', self.state)
        message = ' '.join(f'{key}={value}' for key, value in values.items())
        print(f'[{record["time"]}] {self.method} {event} {message}', flush=True)

    def evaluation(self, label):
        def update(completed, total, scenario_id, ratio):
            if completed == 0 or completed % 10 == 0 or completed == total:
                self.emit('validation_progress', stage=label, completed=completed,
                          total=total, scenario_id=scenario_id, last_ratio=ratio)
        return update

    def __exit__(self, error_type, error, traceback):
        try:
            if error is not None:
                self.emit('failed', status='failed', error=repr(error))
            elif self.state['status'] == 'running':
                self.emit('complete', status='complete')
        finally:
            self.handle.close()


def console_evaluation_progress(label):
    def update(completed, total, scenario_id, ratio):
        if completed == 0 or completed % 10 == 0 or completed == total:
            print(f'{label}: validation={completed}/{total} scenario={scenario_id} ratio={ratio}', flush=True)
    return update
