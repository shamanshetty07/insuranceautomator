"""Fail detection runs abandoned in `running` (crash, kill, lost commit)."""

import sys
from datetime import timedelta


def main() -> int:
    from app.core.config import get_settings
    from app.db.session import get_session_factory
    from app.services.detection_service import fail_stale_runs

    minutes = get_settings().detector_stale_run_minutes
    with get_session_factory()() as db:
        count = fail_stale_runs(db, timedelta(minutes=minutes))
    print(f"Marked {count} detection run(s) with no heartbeat for {minutes} min as failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
