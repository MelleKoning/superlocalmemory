# Security Policy

## SuperLocalMemory V4 Security

### Supported Versions

| Version | Supported |
|:--------|:---------:|
| 4.0.x | Yes |
| 3.8.x | Security fixes only |
| 3.7.x | Security fixes only |
| < 3.7 | No |

### Reporting Vulnerabilities

**Do NOT open public issues for security vulnerabilities.**

Email: admin@superlocalmemory.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Impact assessment
- Suggested fix (if any)

We will respond within 48 hours and provide a fix timeline within 7 days.

### Security Architecture

#### Mode A (Zero-LLM, local core)
- The core store/recall path can run locally at the configured SLM data root
  (default: `~/.superlocalmemory/`)
- The local core has no telemetry, analytics, or phone-home behavior
- Configured remote embedding, remote reranking, connectors, cloud backups, or
  model downloads can make network calls during store/recall or related jobs;
  assess those explicit choices before treating an installation as local-only
- SQLite with WAL mode for data integrity

#### Authentication

**Personal installs (`require_login = false`):** loopback owner is trusted
without a session cookie. The install token is carried as a
personal/API-token header (`X-SLM-Install-Token` / `Authorization:
Bearer …`) and compared with HMAC-SHA256 timing-safe comparison.
Rate limiting: 30 writes/min, 120 reads/min.

**Enterprise installs (`require_login = true`):** the dashboard login
flow issues an `HttpOnly` session cookie (`slm_session`; add `Secure`
with `SLM_DASHBOARD_HTTPS=1`). Mutations require CSRF and origin
controls in addition to the session: `Origin` must match the daemon,
`Sec-Fetch-Site` is checked, and OAuth initiation is same-origin gated
(`src/superlocalmemory/server/routes/backup.py`, `server/origin.py`,
`server/rbac_enforce.py`). See `docs/rbac-teams.md`.

#### Network access (4.1.20+)

**A proxied request is never local.** SuperLocalMemory trusts a caller on
`127.0.0.1` as the local user. It decides that from the socket peer only:
forwarding headers (`X-Forwarded-For`, `Forwarded`, `X-Real-IP` and the like)
are ignored unless you name your proxy in `SLM_TRUSTED_PROXIES`
(comma-separated addresses or networks), and a request that reaches the daemon
from loopback carrying any forwarding header is treated as coming from another
computer. A reverse proxy on the same machine therefore cannot hand its
callers local trust. A same-machine proxy that strips every forwarding header
still looks local; configure the proxy to send `X-Forwarded-For`. See
`docs/distributed-deployment.md`.

#### Data Protection
- Parameterized SQL queries throughout (no SQL injection)
- XSS protection via `escapeHtml()` in all UI rendering
- Security headers: X-Frame-Options, CSP, X-Content-Type-Options
- CORS whitelist with credential control
- Credentials are stored as written (4.1.19+): SuperLocalMemory keeps the
  keys and passwords you save so an agent can recall them. Releases up to
  4.1.18 stripped them on save. Opt-in PII redaction at save
  (`SLM_PII_REDACTION=1`) is unchanged.
- Credential redaction on egress: every request that can carry memory text
  off this machine goes through one gate (`core/outbound_http.py`), which
  redacts credentials for any host that is not loopback. That covers LLM
  providers, cloud embedders, the remote reranker, the Jev answer check, a
  LAN Ollama, and mesh peers. Context injected into an agent session uses
  the same redaction. Requests to loopback do not go through an
  environment proxy. A test fails the build if new code opens an outbound
  request outside the gate. Redaction is pattern-based and best-effort, not
  DLP: a credential with no recognisable shape or label can still pass.
- Remote reranker responses are bounded (8 MiB), redirects are not
  followed, and malformed values suppress bodies without logging them. The
  reranker's HTTPS/non-loopback, userinfo/query/fragment rejections apply to
  the reranker path only; scope SSRF claims to it.

#### Model Supply Chain (Untrusted Checkpoints)

SuperLocalMemory loads local embedding, reranker, and compression models via
Hugging Face. Two risks apply:

- **`trust_remote_code`**: since v3.7.9, `trust_remote_code=True` is passed only
  for an internal allowlist of pinned models (the nomic-embed family, which
  requires custom modeling code). Any other model — including one swapped into
  config through a write path — loads with `trust_remote_code=False` and cannot
  execute repository code at load time.
- **CVE-2025-14926** (`transformers`, SEW tokenizer): arbitrary code execution
  when loading a crafted SEW tokenizer config. No upstream patch exists as of
  2026-07-20. SLM never loads checkpoints automatically; it will be upgraded as
  soon as a patch ships.

Only install models from sources you trust. A malicious checkpoint can execute
code under your user account at load time regardless of these mitigations.

#### Backup and credential-at-rest caveats
- Production backups are independent per-file `sqlite3.backup()` snapshots via
  `BackupManager` (not a coherent cross-store epoch); companion failures are
  non-critical. The coherent `BackupCoordinator` exists as a primitive but is
  not wired to production/routes. Dashboard Export is `memory.db`-only gz.
- Legacy backup destination follows process `umask`, not source-file modes;
  place backups on an encrypted/private volume and verify `0600`/`0700`.
- Stop the daemon (`slm serve stop`) before offline whole-root copy/restore
  so WAL/SHM checkpoint; include sidecars and `lance/` if present.
- Credentials: OS keychain preferred (macOS Keychain / Windows Credential
  Locker / Linux Secret Service); fallback is owner-only plaintext
  `~/.superlocalmemory/.credentials.json` (`0600`, `0700` parent, atomic
  write). Provider/reranker keys persisted in `config.json` are plaintext
  protected only by atomic `0600`; prefer env (`OPENAI_API_KEY`,
  `SLM_CROSS_ENCODER_API_KEY`) to avoid disk persistence.
- Feedback pseudonym: per-install keyed HMAC producing a 16-hex (64-bit)
  `query_hash` ( `learning/feedback.py: _hash_query` ), **not encryption**;
  within-install correlation is possible, key is `0600` beside the DB
  (`.feedback-hash-key`, 32 bytes). Read-only data root falls back to a
  process-local key, losing cross-restart grouping.

#### Migrations (V4.0.0)
- `M038_learning_feedback_channel` (eager) and `M039_scene_fact_members`
  (deferred, after engine tables exist) are automatically applied at startup;
  no manual `slm db migrate` is normally required. `slm db migrate` is
  forward-only (`status`/`--dry-run`/apply) — no rollback. Downgrade requires
  a verified pre-upgrade complete backup (stop daemon, whole-root copy).

#### Compliance
- GDPR Article 15 (right to access): full data export
- GDPR Article 17 (right to erasure): complete erasure including learning data
- EU AI Act data sovereignty: assess the configured deployment and egress path;
  Mode A's local core alone is not a legal or network-locality guarantee
- Tamper-proof audit trail with SHA-256 hash chain
- Bounded / content-free diagnostics export: `slm diagnostics export`

### Dependencies

Run `npm audit` and `pip audit` regularly. Report any findings.

### Research Foundation

The SLM architecture is documented in three public arXiv preprints authored by Varun Pratap Bhardwaj (Qualixar) — these have not undergone external venue review:

- **Paper 1** ([arXiv:2603.02240](https://arxiv.org/abs/2603.02240)): Bayesian trust defense, OWASP-aligned memory poisoning protection
- **Paper 2** ([arXiv:2603.14588](https://arxiv.org/abs/2603.14588)): Information-geometric foundations, cellular sheaf cohomology for contradiction detection
- **Paper 3** ([arXiv:2604.04514](https://arxiv.org/abs/2604.04514)): Trust-weighted forgetting, compliance audit trails, FRQAD mixed-precision integrity

Supporting external work cited by SLM (for example, venues such as ICLR 2026 and Nature Scientific Reports referenced in [ATTRIBUTION.md](ATTRIBUTION.md#optimize-module--research-citations-v36) and LLD-03/LLD-04) is distinct from the three SLM preprints and should be evaluated against its own venue review.

---

Part of [Qualixar](https://qualixar.com) | Author: [Varun Pratap Bhardwaj](https://varunpratap.com)
