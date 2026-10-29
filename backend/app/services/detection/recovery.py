"""Fail detection runs abandoned in `running` (crash, kill, lost commit).

Run:  python -m app.services.detection.recovery
Uses DETECTOR_STALE_RUN_MINUTES (default 30). Not run at app startup: startup must not
need the database, and with several workers a restarting worker would judge runs that
are active elsewhere. See detection_service.fail_stale_runs for the age-only limitation.
"""

import sys
from datetime import timedelta


def main() -> int:
    from app.core.config import get_settings
    from app.db.session import get_session_factory
    from app.services.detection_service import fail_stale_runs

    minutes = get_settings().detector_stale_run_minutes
    with get_session_factory()() as db:
        count = fail_stale_runs(db, timedelta(minutes=minutes))
    print(f"Marked {count} stale running detection run(s) as failed (older than {minutes} min).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
