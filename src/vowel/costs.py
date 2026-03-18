"""Cost tracking and persistence utilities for CodeMode runs."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import logfire
import yaml


class CostManager:
    """Manage generation/run cost records, pricing, and persistence."""

    def __init__(
        self,
        *,
        spec_model: str,
        exploration_model: str,
        generation_id: str | None = None,
        costs_file: Path | None = None,
    ) -> None:
        self.spec_model = spec_model
        self.exploration_model = exploration_model
        self.generation_id = generation_id or self._new_generation_id()
        self._costs_file = (
            costs_file or Path.home() / ".vowel" / "codemode" / "generation_costs.json"
        )
        self._price_table = self._load_costs_yml()
        self._cost_records: dict[str, Any] = self._load_cost_records()
        self._ensure_generation_record()

    @staticmethod
    def _new_generation_id() -> str:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"gen_{ts}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _new_run_id() -> str:
        return f"run_{uuid.uuid4().hex}"

    @staticmethod
    def _default_cost_store() -> dict[str, Any]:
        return {"schema_version": 1, "generations": {}}

    def _load_cost_records(self) -> dict[str, Any]:
        if not self._costs_file.exists():
            return self._default_cost_store()
        try:
            data = json.loads(self._costs_file.read_text(encoding="utf-8"))
        except Exception:
            logfire.warn("Failed to parse cost records, resetting store")
            return self._default_cost_store()

        if not isinstance(data, dict) or "generations" not in data:
            return self._default_cost_store()
        return data

    def _ensure_generation_record(self) -> None:
        generations = self._cost_records.setdefault("generations", {})
        if self.generation_id in generations:
            return
        generations[self.generation_id] = {
            "generation_id": self.generation_id,
            "created_at": datetime.now(UTC).isoformat(),
            "spec_model": self.spec_model,
            "exploration_model": self.exploration_model,
            "totals": {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "requests": 0},
            "runs": {},
        }

    @staticmethod
    def _normalize_models(data: Any) -> dict[str, dict[str, float]]:
        if not isinstance(data, dict):
            return {}

        models_obj = data.get("models")
        normalized: dict[str, dict[str, float]] = {}

        if isinstance(models_obj, dict):
            items = models_obj.items()
        elif isinstance(models_obj, list):
            items = []
            for item in models_obj:
                if isinstance(item, dict):
                    items.extend(item.items())
        else:
            items = []

        for model_name, model_data in items:
            if not isinstance(model_name, str) or not isinstance(model_data, dict):
                continue
            normalized[model_name] = {
                "input_per_million": float(model_data.get("input_per_million", 0.0) or 0.0),
                "output_per_million": float(model_data.get("output_per_million", 0.0) or 0.0),
                "cached_input_per_million": float(
                    model_data.get("cached_input_per_million", 0.0) or 0.0
                ),
            }

        return normalized

    def _load_costs_yml(self) -> dict[str, Any]:
        candidates = [Path.cwd() / "costs.yml", Path(__file__).resolve().parents[2] / "costs.yml"]
        for path in candidates:
            if not path.exists():
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            models = self._normalize_models(data)
            if models:
                return {"models": models}
        return {}

    def _persist_costs_atomic(self) -> None:
        self._costs_file.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._cost_records, ensure_ascii=False, indent=2) + "\n"
        lock_path = self._costs_file.parent / ".generation_costs.lock"

        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self._costs_file.parent, delete=False
                ) as tmp:
                    tmp.write(payload)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    tmp_path = Path(tmp.name)
                os.replace(tmp_path, self._costs_file)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _ensure_run_record(self, run_id: str, func_name: str) -> None:
        generation = self._cost_records["generations"][self.generation_id]
        runs = generation.setdefault("runs", {})
        if run_id in runs:
            return
        runs[run_id] = {
            "run_id": run_id,
            "func_name": func_name,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "error": None,
            "steps": {},
            "totals": {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "requests": 0},
        }

    def _get_run_record(self, run_id: str) -> dict[str, Any]:
        return self._cost_records["generations"][self.generation_id]["runs"][run_id]

    @staticmethod
    def _run_usage_dict(usage: Any) -> dict[str, int]:
        return {
            "requests": int(getattr(usage, "requests", 0) or 0),
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cached_input_tokens": int(getattr(usage, "cached_input_tokens", 0) or 0),
        }

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        normalized = model_name.strip()
        if ":" in normalized:
            normalized = normalized.split(":", 1)[1]
        if "/" in normalized:
            normalized = normalized.rsplit("/", 1)[1]
        return normalized

    def _resolve_price_from_costs_yml(self, model_name: str) -> dict[str, float] | None:
        models = self._price_table.get("models") if isinstance(self._price_table, dict) else None
        if not isinstance(models, dict):
            return None

        normalized = self._normalize_model_name(model_name)
        for key in (model_name, normalized):
            data = models.get(key)
            if not isinstance(data, dict):
                continue
            return {
                "input_per_million": float(data.get("input_per_million", 0.0) or 0.0),
                "output_per_million": float(data.get("output_per_million", 0.0) or 0.0),
                "cached_input_per_million": float(data.get("cached_input_per_million", 0.0) or 0.0),
            }
        return None

    def _resolve_price(self, model_name: str) -> tuple[dict[str, float] | None, str, bool]:
        normalized = self._normalize_model_name(model_name)

        try:
            import genai_prices  # type: ignore

            for attr in ("get_price", "lookup_price", "resolve_price"):
                fn = getattr(genai_prices, attr, None)
                if callable(fn):
                    for name in (model_name, normalized):
                        out = fn(name)
                        if isinstance(out, dict):
                            return (
                                {
                                    "input_per_million": float(
                                        out.get("input_per_million")
                                        or out.get("input")
                                        or out.get("prompt")
                                        or 0.0
                                    ),
                                    "output_per_million": float(
                                        out.get("output_per_million")
                                        or out.get("output")
                                        or out.get("completion")
                                        or 0.0
                                    ),
                                    "cached_input_per_million": float(
                                        out.get("cached_input_per_million")
                                        or out.get("cached_input")
                                        or 0.0
                                    ),
                                },
                                "genai-prices",
                                False,
                            )
        except Exception:
            pass

        yml_price = self._resolve_price_from_costs_yml(model_name)
        if yml_price is not None:
            return yml_price, "costs.yml", False

        return None, "missing", True

    def _estimate_step_usd(self, model_name: str, usage: dict[str, int]) -> tuple[float, str, bool]:
        price, source, missing = self._resolve_price(model_name)
        if price is None:
            return 0.0, source, True

        in_cost = usage["input_tokens"] / 1_000_000 * price["input_per_million"]
        out_cost = usage["output_tokens"] / 1_000_000 * price["output_per_million"]
        cached_cost = usage["cached_input_tokens"] / 1_000_000 * price["cached_input_per_million"]
        return in_cost + out_cost + cached_cost, source, missing

    def _recompute_totals(self) -> None:
        generation = self._cost_records["generations"][self.generation_id]
        g_totals = {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "requests": 0}

        for run in generation.get("runs", {}).values():
            r_totals = {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "requests": 0}
            for step in run.get("steps", {}).values():
                usages = step.get("usages", [])
                for item in usages:
                    usage = item.get("usage", {})
                    r_totals["usd"] += float(item.get("usd", 0.0) or 0.0)
                    r_totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                    r_totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
                    r_totals["requests"] += int(usage.get("requests", 0) or 0)
            run["totals"] = r_totals

            g_totals["usd"] += r_totals["usd"]
            g_totals["input_tokens"] += r_totals["input_tokens"]
            g_totals["output_tokens"] += r_totals["output_tokens"]
            g_totals["requests"] += r_totals["requests"]

        generation["totals"] = g_totals

    def start_run(self, *, run_id: str | None, func_name: str) -> str:
        final_run_id = run_id or self._new_run_id()
        self._ensure_generation_record()
        self._ensure_run_record(final_run_id, func_name)
        self._persist_costs_atomic()
        return final_run_id

    def record_agent_usage(
        self, *, run_id: str, step_key: str, result: Any, model_name: str
    ) -> None:
        run = self._get_run_record(run_id)
        step = run.setdefault("steps", {}).setdefault(step_key, {"usages": []})

        usage_obj = result.usage() if callable(getattr(result, "usage", None)) else None
        usage = (
            self._run_usage_dict(usage_obj) if usage_obj is not None else self._run_usage_dict(None)
        )
        usd, price_source, price_missing = self._estimate_step_usd(model_name, usage)

        step_item = {
            "timestamp": datetime.now(UTC).isoformat(),
            "agent_run_id": getattr(result, "run_id", None),
            "model_name": model_name,
            "usage": usage,
            "usd": usd,
            "price_source": price_source,
            "price_missing": price_missing,
        }
        step["usages"].append(step_item)

        self._recompute_totals()
        self._persist_costs_atomic()

        logfire.info(
            "CodeMode step cost recorded",
            generation_id=self.generation_id,
            run_id=run_id,
            step=step_key,
            model_name=model_name,
            usd=usd,
            usage=usage,
            price_source=price_source,
            price_missing=price_missing,
        )

    def mark_run_completed(self, run_id: str) -> None:
        run_rec = self._get_run_record(run_id)
        run_rec["status"] = "completed"
        run_rec["completed_at"] = datetime.now(UTC).isoformat()
        self._recompute_totals()
        self._persist_costs_atomic()

    def mark_run_failed(self, run_id: str, error: str) -> None:
        run_rec = self._get_run_record(run_id)
        run_rec["status"] = "failed"
        run_rec["error"] = error
        run_rec["completed_at"] = datetime.now(UTC).isoformat()
        self._recompute_totals()
        self._persist_costs_atomic()

    def print_total_cost(self, run_id: str | None = None) -> None:
        generation = self._cost_records["generations"].get(self.generation_id, {})
        if run_id is not None:
            run = generation.get("runs", {}).get(run_id)
            if not run:
                print(f"run not found: {run_id}")
                return
            totals = run.get("totals", {})
            print(
                "run_cost",
                run_id,
                f"usd={totals.get('usd', 0.0):.6f}",
                f"input_tokens={totals.get('input_tokens', 0)}",
                f"output_tokens={totals.get('output_tokens', 0)}",
                f"requests={totals.get('requests', 0)}",
            )
            return

        totals = generation.get("totals", {})
        print(
            "generation_cost",
            self.generation_id,
            f"usd={totals.get('usd', 0.0):.6f}",
            f"input_tokens={totals.get('input_tokens', 0)}",
            f"output_tokens={totals.get('output_tokens', 0)}",
            f"requests={totals.get('requests', 0)}",
        )
