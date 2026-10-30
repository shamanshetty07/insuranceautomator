import pytest

from tests.db_safety import disposable_database_problem


@pytest.mark.parametrize("url", [
    "postgresql+psycopg://u:p@localhost:5432/insurance_test",
    "postgresql+psycopg://u:p@127.0.0.1/test_db",
    "postgresql+psycopg://u:p@[::1]:5433/TESTING",
    "postgresql+psycopg:///insurance_test",
    "postgresql://u@localhost/x_test_y",
])
def test_local_disposable_urls_are_allowed(url):
    assert disposable_database_problem(url) is None


@pytest.mark.parametrize("url,needle", [
    ("postgresql+psycopg://u:p@ep-cool-123.eu-central-1.aws.neon.tech/neondb_test", "localhost"),
    ("postgresql+psycopg://u:p@db.example.com/insurance_test", "localhost"),
    ("postgresql+psycopg://u:p@192.168.1.5/insurance_test", "localhost"),
    ("postgresql+psycopg://u:p@localhost/insurance", "'test'"),
    ("postgresql+psycopg://u:p@localhost/", "'test'"),
    ("postgresql+psycopg:///insurance_test?host=db.example.com", "query"),
    ("sqlite:///insurance_test.db", "PostgreSQL"),
    ("not a url", "valid"),
])
def test_other_urls_are_refused(url, needle):
    problem = disposable_database_problem(url)
    assert problem and needle in problem


def test_refusal_message_never_echoes_the_url():
    url = "postgresql+psycopg://user:s3cret@remote.example.com/insurance_test"
    assert "s3cret" not in disposable_database_problem(url)
