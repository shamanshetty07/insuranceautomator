# Running the database tests safely

The 197 database tests (and any added since) are **skipped** unless `TEST_DATABASE_URL` is set. They have
never been run in this project's recent sprints, so everything they cover is unverified.

## How the tests choose a database

`backend/tests/conftest.py`:

- reads `TEST_DATABASE_URL` from the **process environment only** (never `backend/.env`);
- refuses to start if it equals `DATABASE_URL` in the environment;
- refuses to start unless `tests/db_safety.py` accepts it: PostgreSQL, host `localhost`/`127.0.0.1`/`::1` or a
  local socket, a database name containing `test`, and no host override in query parameters. The error
  message never repeats the URL;
- forces the app's `DATABASE_URL` to that URL, or to an unroutable address when it is unset, before any app
  import, so no test can reach a configured database by accident;
- in the session fixture `db_engine`: **drops and recreates the `public` schema** and runs every Alembic
  migration to head. Each test then truncates `inspections` (cascading).

Migration tests (`test_migrations.py`, `test_verification_migration.py`, `test_report_claims_migration.py`)
also downgrade and upgrade that database.

**Never point `TEST_DATABASE_URL` at Neon, a shared database, or one with data you want.** The guard blocks
remote hosts, but a local database whose name contains `test` is still wiped.

## A disposable local database (needs your approval; not done)

Option A, Docker (nothing touches the existing local PostgreSQL on port 5432):

```bash
docker run --rm -d --name insurance-test-db -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=insurance_test -p 55432:5432 postgres:16
export TEST_DATABASE_URL=postgresql+psycopg://test:test@127.0.0.1:55432/insurance_test
cd backend && source .venv/bin/activate && pytest -q
docker stop insurance-test-db   # --rm deletes it
```

This pulls the `postgres:16` image (a download) and needs Docker.

Option B, the existing local PostgreSQL on `127.0.0.1:5432`: it requires a password nobody has
provided. Creating a role and an `insurance_test` database in it needs that password and is a change to
a server other people may use.

Approvals needed before either option:

1. Permission to pull the PostgreSQL image (A), or credentials and permission to create a database (B).
2. Permission to run migrations and destructive tests **against that disposable database only**.
3. Acknowledgement that Neon stays untouched, and that the database's current schema is irrelevant
   because the tests rebuild it from the repository migrations.

## What remains unverified until then

- Every Alembic migration (0001–0006), including upgrade/downgrade with data.
- The relational report and claim persistence, foreign keys, cascades, deferred constraints and
  `image_in_use`.
- `associate_claims`, `load_prior_verdicts` and `assess_claim_quality` against real rows.
- Every API test using the `client` fixture.
