# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What the hosted answer check may never receive, and what it must still read.

Two directions, both measured on every run:

* every fake credential below, in a shape real keys and passwords take, is
  replaced before text leaves the machine;
* ordinary text a memory needs to answer a question — a commit id, a UUID used
  as an identifier, a checksum in a changelog, a port, token counts — passes
  through byte for byte.

Every value here is an obvious placeholder (sequential hex, the word FAKE).
They are assembled at run time so no literal in this file has the exact shape
a repository secret scanner looks for.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.security_primitives import redact_secrets
from superlocalmemory.retrieval.hosted_redaction import (
    REDACTED_MARKER,
    redact_for_hosted_judge,
)


def _j(*parts: str) -> str:
    return "".join(parts)


HEX32 = "0123456789abcdef" * 2
HEX40 = HEX32 + "01234567"
HEX64 = HEX32 * 2
UUID = "01234567-89ab-cdef-0123-456789abcdef"
B64_40 = "FAKEfake/FAKEfake+FAKEfake0123456789abcd"
PEM_BODY = "MIIEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFA"

#: (shape, text as a memory might hold it, the value that must not survive)
FAKE_CREDENTIALS: tuple[tuple[str, str, str], ...] = (
    ("hex key after an env label", f"DD_API_KEY={HEX32}", HEX32),
    ("40-hex token after token:", f"token: {HEX40}", HEX40),
    ("64-hex secret after secret=", f"secret={HEX64}", HEX64),
    ("UUID key after an env label", f"HEROKU_API_KEY={UUID}", UUID),
    ("UUID key in prose", f"my Heroku API key is {UUID}", UUID),
    ("hex key in prose", f"the Datadog api key {HEX32} works", HEX32),
    ("postgres URL password",
     "DATABASE_URL=postgres://admin:FakePass123@db1.corp.internal:5432/app",
     "FakePass123"),
    ("URL password containing @", "mysql://root:p@ssw0rdFAKE@localhost/db", "p@ssw0rdFAKE"),
    ("URL password with empty user", "redis://:FakeRedisPass@cache.internal:6379/0",
     "FakeRedisPass"),
    ("URL token as password",
     "https://x-access-token:FAKEtoken1234567890@github.com/org/repo.git",
     "FAKEtoken1234567890"),
    ("URL token as user only",
     _j("https://", "FAKEFAKE0123456789FAKEFAKE", "@github.com/org/repo.git"),
     "FAKEFAKE0123456789FAKEFAKE"),
    ("mongodb+srv URL", "mongodb+srv://appuser:Fake%40Pass9@cluster0.example.net/",
     "Fake%40Pass9"),
    ("password=", "password=hunter2FAKE", "hunter2FAKE"),
    ("JSON password", '{"password": "correct-horse-FAKE-staple"}', "correct-horse-FAKE-staple"),
    ("YAML db_password", "db_password: S3cr3tFake!", "S3cr3tFake!"),
    ("env secret access key", f"export AWS_SECRET_ACCESS_KEY={B64_40}", B64_40),
    ("password in prose", "my password is hunter2FAKE", "hunter2FAKE"),
    ("passphrase in prose", "the wifi passphrase is Tr0ub4dor&3", "Tr0ub4dor&3"),
    ("Slack user token", _j("xo", "xp-", "0000000000-0000000000-", HEX32), HEX32),
    ("Slack app-level token", _j("xo", "xa-2-", "0000000000-", HEX32), HEX32),
    ("Slack refresh token", _j("xo", "xr-", "0000000000-", HEX32), HEX32),
    ("Slack session token", _j("xo", "xs-", "0000000000-", HEX32), HEX32),
    ("Slack app token", _j("xa", "pp-1-", "A0000000000-", HEX32), HEX32),
    ("Slack webhook URL",
     _j("https://hooks.slack.com/services/", "T00000000/B00000000/", "FAKEFAKEFAKEFAKEFAKEFAKE"),
     "FAKEFAKEFAKEFAKEFAKEFAKE"),
    ("Stripe live secret key", _j("sk", "_live_", "FAKEFAKEFAKEFAKEFAKEFAKE"),
     "FAKEFAKEFAKEFAKEFAKEFAKE"),
    ("Stripe test secret key", _j("sk", "_test_", "FAKEFAKEFAKEFAKEFAKEFAKE"),
     "FAKEFAKEFAKEFAKEFAKEFAKE"),
    ("Stripe restricted key", _j("rk", "_live_", "FAKEFAKEFAKEFAKEFAKEFAKE"),
     "FAKEFAKEFAKEFAKEFAKEFAKE"),
    ("Stripe webhook secret", _j("wh", "sec_", "FAKEFAKEFAKEFAKEFAKEFAKE"),
     "FAKEFAKEFAKEFAKEFAKEFAKE"),
    ("short key after a label", "api_key=abc12345", "abc12345"),
    ("short token after a label", "token=Zx9_short", "Zx9_short"),
    ("short Bearer token", "Authorization: Bearer abc123def456", "abc123def456"),
    ("Basic auth header", "Authorization: Basic dXNlcjpGQUtFcGFzcw==", "dXNlcjpGQUtFcGFzcw=="),
    ("PEM private key body",
     _j("-----BEGIN RSA ", "PRIVATE KEY-----\n", PEM_BODY, "\n", PEM_BODY,
        "\n-----END RSA ", "PRIVATE KEY-----"),
     PEM_BODY),
    ("GitHub user-to-server token", _j("gh", "u_", "FAKE" * 9), "FAKE" * 9),
    ("GitHub refresh token", _j("gh", "r_", "FAKE" * 9), "FAKE" * 9),
    ("GitLab token", _j("gl", "pat-", "FAKEFAKEFAKEFAKE0123"), "FAKEFAKEFAKEFAKE0123"),
    ("npm token", _j("np", "m_", "FAKE" * 9), "FAKE" * 9),
    ("Hugging Face token", _j("h", "f_", "FAKE" * 8, "ab"), "FAKE" * 8 + "ab"),
    ("SendGrid key", _j("S", "G.", "FAKEFAKEFAKEFAKEFAKE12", ".", "FAKE" * 10, "ABC"),
     "FAKE" * 10 + "ABC"),
    ("Mailgun key", _j("ke", "y-", HEX32), HEX32),
    ("Telegram bot token", _j("123456789", ":AA", "FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE1"),
     "FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE1"),
    ("Google OAuth client secret", _j("GOC", "SPX-", "FAKEFAKEFAKEFAKEFAKE1234"),
     "FAKEFAKEFAKEFAKEFAKE1234"),
    ("Azure storage AccountKey",
     _j("DefaultEndpointsProtocol=https;AccountName=fakeacct;AccountKey=", "FAKE" * 21, "==;"),
     "FAKE" * 21),
    ("CLI password flag", "psql --password FakePass1 -h db1", "FakePass1"),
    ("client_secret=", "client_secret=FAKEclientsecret", "FAKEclientsecret"),
    ("PGPASSWORD env", "PGPASSWORD=fakepw99 psql", "fakepw99"),
    ("npmrc auth token", "//registry.npmjs.org/:_authToken=FAKE-auth-token-0000",
     "FAKE-auth-token-0000"),
    ("libpq key=value DSN", "host=db1 user=app password=FakePw77 dbname=x", "FakePw77"),
    ("Django SECRET_KEY", "SECRET_KEY = 'django-insecure-FAKEfakeFAKEfake'",
     "django-insecure-FAKEfakeFAKEfake"),
    # Shapes the scanner already caught before this change — kept so a
    # regression in the old coverage shows up in the same number.
    ("OpenRouter key", _j("sk", "-or-v1-", HEX64), HEX64),
    ("AWS access key id", _j("AK", "IA", "FAKEFAKEFAKEFAKE"), "FAKEFAKEFAKEFAKE"),
    ("GitHub classic PAT", _j("gh", "p_", "FAKE" * 9), "FAKE" * 9),
)

#: (kind of information, text as a memory might hold it) — must pass unchanged.
ORDINARY_TEXT: tuple[tuple[str, str], ...] = (
    ("git commit SHA", f"Fixed in commit {HEX40}; see the PR."),
    ("short commit id", "Reverted 26b261d1 after the release check."),
    ("UUID as a record id", f"Session {UUID} closed at 10:42."),
    ("UUID as a request id", f"request_id={UUID} returned 404"),
    ("sha256 in a changelog", f"sha256: {HEX64}  superlocalmemory-4.1.18.tar.gz"),
    ("md5 checksum", f"md5 {HEX32} matches the download"),
    ("port number", "The daemon listens on port 8765 and the dashboard on 8766."),
    ("localhost URL", "Open http://localhost:8765/dashboard to check."),
    ("URL with a port, no credentials", "Postgres is at postgres://db1.corp.internal:5432/app"),
    ("model token limit", "max_tokens=512 and num_tokens: 1024 keep it short"),
    ("token count", "token_count: 1500 for that prompt"),
    ("tokenizer name", "tokenizer: bert-base-multilingual-cased"),
    ("primary key", "primary_key: id, sort_key: created_at"),
    ("password field in prose", "The password field is required on the login form."),
    ("password policy in prose", "Our password policy needs 12 characters."),
    ("secret in prose", "The secret to good pasta is salting the water."),
    ("API key rotation in prose", "API key rotation is due on Friday."),
    ("token in prose", "Each token costs a fraction of a cent."),
    ("email address", "Mail alice@example.com about the launch."),
    ("file path", "Config lives in ~/.superlocalmemory/config.json on this Mac."),
    ("version string", "Upgraded from 4.1.17 to 4.1.18 last night."),
    ("password is a word", "Her password is stored in the keychain."),
    ("secret: true", "secret: true marks the field in the schema"),
)


@pytest.mark.parametrize(
    ("shape", "text", "value"), FAKE_CREDENTIALS, ids=[c[0] for c in FAKE_CREDENTIALS],
)
def test_a_fake_credential_never_leaves_the_machine(shape: str, text: str, value: str) -> None:
    out = redact_for_hosted_judge(text)
    assert value not in out, f"{shape}: credential survived: {out!r}"
    assert REDACTED_MARKER in out


@pytest.mark.parametrize(
    ("kind", "text"), ORDINARY_TEXT, ids=[o[0] for o in ORDINARY_TEXT],
)
def test_ordinary_information_reaches_the_answer_check_unchanged(kind: str, text: str) -> None:
    assert redact_for_hosted_judge(text) == text, kind


def test_the_label_of_a_redacted_value_is_kept() -> None:
    """The memory still says a password exists — only the value is gone."""
    out = redact_for_hosted_judge("db_password: S3cr3tFake!")
    assert out == f"db_password: {REDACTED_MARKER}"


def test_the_host_of_a_connection_url_is_kept() -> None:
    out = redact_for_hosted_judge("postgres://admin:FakePass123@db1.corp.internal:5432/app")
    assert out == f"postgres://admin:{REDACTED_MARKER}@db1.corp.internal:5432/app"


def test_a_short_secret_keeps_none_of_its_characters_in_the_local_marker() -> None:
    """The local marker keeps four characters for logs; for a short password
    that is most of it, so a short labelled value keeps none."""
    out = redact_secrets("password=hunter2FAKE", aggression="high")
    assert "FAKE" not in out
    assert "hunter2" not in out


def test_normal_aggression_is_unchanged_for_stored_content() -> None:
    """What is stored is scrubbed at normal aggression; a local memory such as
    a wifi password must stay answerable on this machine."""
    for text in ("password=hunter2FAKE", "my password is hunter2FAKE",
                 f"DD_API_KEY={HEX32}", "postgres://admin:FakePass123@db1/app"):
        assert redact_secrets(text) == text


def test_redaction_stays_linear_on_hostile_input() -> None:
    import time

    hostile = ("password=" * 4000) + ("a:" * 8000) + ("x" * 20000) + "@" + ("token " * 4000)
    started = time.perf_counter()
    redact_for_hosted_judge(hostile)
    assert time.perf_counter() - started < 1.0
