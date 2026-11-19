"""Persistent USD25 reservation guard. Does not call any provider.

Every attempt needs a unique ID and a conservative maximum charge including
input/output/caching/provider overhead. Unknown outcomes retain the full
reservation. Reopening a ledger never releases outstanding reservations.
"""
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
import fcntl, json, os, tempfile

CAP = Decimal("25")

class BudgetLedger:
    def __init__(self, path): self.path = Path(path)

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {"cap_usd": "25", "attempts": {}}
            if Decimal(state["cap_usd"]) != CAP: raise ValueError("budget cap changed")
            yield state
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as tmp:
                json.dump(state, tmp, indent=2); tmp.write("\n"); tmp.flush(); os.fsync(tmp.fileno())
            os.replace(tmp.name, self.path)

    @staticmethod
    def committed(state):
        return sum((Decimal(v.get("actual_usd", v["maximum_usd"])) for v in state["attempts"].values()), Decimal(0))

    def reserve(self, attempt_id, maximum_usd):
        amount = Decimal(str(maximum_usd))
        if not amount.is_finite() or amount <= 0: raise ValueError("invalid maximum")
        with self.transaction() as state:
            if attempt_id in state["attempts"]: raise ValueError("attempt already reserved; never resend on resume")
            if self.committed(state) + amount > CAP: raise RuntimeError("USD25 budget exhausted")
            state["attempts"][attempt_id] = {"maximum_usd": str(amount), "status": "reserved"}

    def settle(self, attempt_id, actual_usd=None):
        with self.transaction() as state:
            row = state["attempts"][attempt_id]
            if row["status"] != "reserved": raise ValueError("attempt already settled")
            actual = Decimal(row["maximum_usd"] if actual_usd is None else str(actual_usd))
            if not actual.is_finite() or actual < 0 or actual > Decimal(row["maximum_usd"]):
                raise ValueError("actual cost outside reservation; halt and reconcile")
            row.update(actual_usd=str(actual), status="unknown_charged_at_maximum" if actual_usd is None else "settled")
