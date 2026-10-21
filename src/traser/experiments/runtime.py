"""Single-owner guard for complete GPU jobs in this release's output tree."""

from contextlib import contextmanager
from functools import wraps
import os, sqlite3
from traser.io import ROOT


@contextmanager
def experiment_lock():
    path = ROOT / "outputs/runtime/owner.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=1)
    try:
        try:
            db.execute("CREATE TABLE IF NOT EXISTS owner(pid INTEGER)")
            db.commit()
            db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise RuntimeError("Another complete experiment owns the release output tree") from exc
        db.execute("DELETE FROM owner")
        db.execute("INSERT INTO owner VALUES (?)", (os.getpid(),))
        try:
            yield
        finally:
            db.rollback()
    finally:
        db.close()


def exclusive_experiment(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        if kwargs.get("check_only") or len(args) > 2 and args[2]:
            return function(*args, **kwargs)
        with experiment_lock():
            return function(*args, **kwargs)

    return guarded
