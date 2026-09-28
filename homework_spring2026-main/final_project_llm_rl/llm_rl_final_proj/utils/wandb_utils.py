from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


class WandBLogger:
    def __init__(
        self,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        enabled: bool = True,
        local_dir: str | Path | None = None,
    ):
        self.enabled = enabled
        self._run = None
        self._wandb = None
        self._metrics_fh = None
        self._local_dir = Path(local_dir) if local_dir is not None else None
        if self._local_dir is not None:
            self._local_dir.mkdir(parents=True, exist_ok=True)
            (self._local_dir / "config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            self._metrics_fh = (self._local_dir / "metrics.jsonl").open("a", encoding="utf-8")
        if self.enabled:
            try:
                import wandb  # type: ignore
            except Exception:
                self.enabled = False
                return
            self._wandb = wandb
            self._run = wandb.init(project=project, name=run_name, config=config)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._metrics_fh is not None:
            filtered = _filter_metrics_for_local_jsonl(metrics)
            if filtered:
                record = {"step": step, "metrics": filtered}
                self._metrics_fh.write(json.dumps(record, sort_keys=True) + "\n")
                self._metrics_fh.flush()
        if (not self.enabled) or (self._wandb is None):
            return
        self._wandb.log(metrics, step=step)

    def log_table(self, name: str, rows: List[Dict[str, Any]], step: Optional[int] = None) -> None:
        if (not self.enabled) or (self._wandb is None) or (not rows):
            return
        columns = list(rows[0].keys())
        for row in rows[1:]:
            for k in row.keys():
                if k not in columns:
                    columns.append(k)
        table = self._wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row.get(c) for c in columns])
        self._wandb.log({name: table}, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
        if self._metrics_fh is not None:
            self._metrics_fh.close()
            self._metrics_fh = None


def _filter_metrics_for_local_jsonl(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("samples/"):
            continue
        if isinstance(value, bool):
            out[key] = value
            continue
        if isinstance(value, int):
            out[key] = value
            continue
        if isinstance(value, float):
            out[key] = value if math.isfinite(value) else str(value)
            continue
        if isinstance(value, str):
            out[key] = value
            continue
        if value is None:
            out[key] = None
    return out
