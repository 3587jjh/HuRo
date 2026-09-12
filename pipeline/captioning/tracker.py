from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict


@dataclass
class RejectionEvent:
    reason: str
    clip_id: str = ""
    window: str = ""
    detail: str = ""
    think: str = ""


class RejectionTracker:
    def __init__(self, enabled: bool = False, max_examples: int = 3):
        self.enabled = enabled
        self.max_examples = max_examples
        self.counts: Dict[str, int] = defaultdict(int)
        self.examples: Dict[str, List[RejectionEvent]] = defaultdict(list)
        self.total_windows: int = 0
        self.total_passed: int = 0
        self.selection_log: List[dict] = []
        self.window_logs: List[dict] = []

    def reject(self, reason: str, **kwargs) -> str:
        """Record rejection event. Returns 'n/a'."""
        if self.enabled:
            self.counts[reason] += 1
            if len(self.examples[reason]) < self.max_examples:
                self.examples[reason].append(RejectionEvent(reason=reason, **kwargs))
        return "n/a"

    def passed(self):
        """Record a passed window."""
        if self.enabled:
            self.total_passed += 1

    def log_selection(self, clip_id: str = "", window: str = "",
                      field_winners: dict = None, reject_counts: dict = None):
        """Log multi-sample selection decision.
        field_winners: {field: winner_meta_or_None}. reject_counts: {reason: count}."""
        if not self.enabled:
            return
        entry = {
            "clip_id": clip_id,
            "window": window,
            "field_winners": field_winners,
            "n_fields_active": sum(
                1 for v in (field_winners or {}).values()
                if v is not None and v.get("text") != "n/a"
            ),
        }
        if reject_counts:
            entry["rejects"] = reject_counts
        self.selection_log.append(entry)

    def log_window(self, entry: dict):
        """Append a per-window detailed log entry."""
        if self.enabled:
            self.window_logs.append(entry)

    def window_start(self):
        """Call at the start of each window to count total."""
        if self.enabled:
            self.total_windows += 1

    def to_dict(self, clip_id: str = "") -> dict:
        """Export tracker state as a JSON-serializable dict for aggregation."""
        d = {
            "clip_id": clip_id,
            "total_windows": self.total_windows,
            "total_passed": self.total_passed,
            "total_rejected": self.total_windows - self.total_passed,
            "counts": dict(self.counts),
            "examples": {
                reason: [
                    {"window": ev.window,
                     "detail": ev.detail, "think": ev.think[:120]}
                    for ev in evs
                ]
                for reason, evs in self.examples.items()
            },
        }
        if self.selection_log:
            fields_dist: Dict[int, int] = defaultdict(int)
            for sel in self.selection_log:
                n = sel.get("n_fields_active", 0)
                fields_dist[n] += 1
            d["selection_fields_dist"] = {str(k): v for k, v in sorted(fields_dist.items())}
            d["selection_details"] = self.selection_log
        if self.window_logs:
            d["window_logs"] = self.window_logs
        return d

    def reset(self):
        """Clear for next clip."""
        self.counts.clear()
        self.examples.clear()
        self.total_windows = 0
        self.total_passed = 0
        self.selection_log.clear()
        self.window_logs.clear()
