# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Labelled (source, derived fact, expected finding) pairs for the source check.

Synthetic and de-identified; no private memory text. ``None`` means the fact
is faithful and must not be flagged. Hard cases are included on purpose — a
faithful paraphrase that drops a negation word ("is not enabled" -> "is
disabled"), damage whose ordered items sit behind a verb — so the measured
precision and recall are honest, not tuned.
"""

from __future__ import annotations

KESTREL = "The Kestrel checkpoint was recalled at rank 1 in 2004.6 ms."
POLICY = "Never publish without approval from the release owner."
ORDER = "Ship the iOS build before the Android build."
STATUS = "Previously the team used Jenkins, currently it uses GitHub Actions."

FAITHFUL: list[tuple[str, str]] = [
    # negation said another way (the known hard cases)
    ("The cache is not enabled.", "The cache is disabled."),
    ("We do not allow guest accounts.", "Guest accounts are disallowed."),
    ("The job never runs on weekends.", "The job runs only on weekdays."),
    ("The service does not support IPv6.", "IPv6 is unsupported by the service."),
    ("No tests failed in the nightly run.", "All tests passed in the nightly run."),
    ("The feature is not available on mobile.", "The feature is unavailable on mobile."),
    ("Do not deploy without a review.", "Every deploy requires a review."),
    ("The bucket is not public.", "The bucket is private."),
    # units and number formats
    ("Recall took 3.0 s at p95.", "Recall took 3 seconds at p95."),
    ("The import ran for 90 min.", "The import ran for 90 minutes."),
    ("The model uses 4.5 GB of memory.", "The model needs 4.5 GB of RAM."),
    ("The page loads in 250 ms.", "Page load time is 250 milliseconds."),
    ("Error rate dropped to 0.5%.", "The error rate fell to 0.5 %."),
    ("The plan costs $12.50 per month.", "The monthly plan is $12.5."),
    ("We store 1,200 records per batch.", "Each batch holds 1200 records."),
    ("The API listens on port 8080.", "Port 8080 is where the API listens."),
    ("Upgrade to v2.3.1 before Friday.", "The upgrade target is 2.3.1, due before Friday."),
    ("Bug #412 is fixed.", "Issue #412 has been fixed."),
    ("Throughput reached 1,500 req/s.", "Throughput hit 1500 requests per second."),
    ("The build took 12.5 minutes.", "Building took 12.5 min."),
    (KESTREL, "The Kestrel checkpoint was recalled at rank 1 in 2004.6 milliseconds."),
    (KESTREL, "The Kestrel checkpoint recall took 2,004.6 ms."),
    # relative dates resolved
    ("I renewed the licence yesterday.", "Dana renewed the licence on 2026-10-05."),
    ("The demo is next Tuesday.", "The demo is on 2026-10-13."),
    ("We shipped the fix 3 days ago.", "The fix shipped on 2026-10-03."),
    ("The audit happens next month.", "The audit happens in 2026-11."),
    ("Last year the team moved to Lisbon.", "In 2025 the team moved to Lisbon."),
    ("The review is tomorrow.", "The review is on 2026-10-07."),
    ("Last week we froze the schema.", "The schema was frozen on 2026-09-29."),
    # dates reformatted
    ("The launch is on 6 Oct 2026.", "The launch is on 2026-10-06."),
    ("Kickoff was March 15, 2026.", "Kickoff happened on 2026-03-15."),
    ("The contract ends 31/12/2026.", "The contract ends on 2026-12-31."),
    ("The freeze starts in June 2026.", "The code freeze begins in June 2026."),
    ("Payment is due on 2026-04-30.", "Payment is due April 30, 2026."),
    ("The offsite is 12.11.2026.", "The offsite is on 2026-11-12."),
    ("The p95 was 2019.3 ms on 6 Oct 2026.", "On 2026-10-06 the p95 was 2019.3 ms."),
    ("Version 4.1.21 shipped on 2021-04-01.", "Release 4.1.21 shipped on 2021-04-01."),
    ("Score 2020.5 was reached in May 2020.", "The score reached 2020.5 in May 2020."),
    # order kept
    (ORDER, "The Android build ships after the iOS build."),
    (ORDER, "The iOS build ships before the Android build."),
    ("Run the migration before the deploy.", "The migration runs ahead of the deploy."),
    ("Back up the database before the upgrade.", "A database backup comes before the upgrade."),
    ("Write tests before the refactor.", "Tests are written prior to the refactor."),
    # previously / currently kept
    (STATUS, "The team uses GitHub Actions."),
    (STATUS, "The team previously used Jenkins."),
    ("We formerly hosted on Heroku; now we host on Fly.", "Hosting is on Fly."),
    ("The API used to return XML, but now it returns JSON.", "The API returns JSON."),
    ("Earlier the limit was 100, currently it is 500.", "The current limit is 500."),
    # plain paraphrases
    (POLICY, "Publishing must never happen without the release owner's approval."),
    ("The nightly job does not retry failed uploads.",
     "Failed uploads are not retried by the nightly job."),
    ("The cluster has 12 nodes.", "There are 12 nodes in the cluster."),
    ("Alice prefers Postgres over MySQL.", "Alice likes Postgres more than MySQL."),
    ("The Pune warehouse ships 340 orders a day.", "The Pune warehouse ships 340 orders daily."),
    ("Commit 8f3a2b1 fixed the leak in 3s.", "The leak fix in commit 8f3a2b1 took 3s."),
    ("The SLA is 99.9% uptime.", "Uptime SLA: 99.9%."),
    ("Version 4.1.21 added kind filters.", "Kind filters arrived in version 4.1.21."),
    ("No, the build uses GitHub Actions.", "The build uses GitHub Actions."),
    ("We do not use Jenkins; we use GitHub Actions.", "The team uses GitHub Actions."),
    ("Meetings are on Mondays at 10.", "The team meets on Mondays at 10."),
    ("The report covers 2025.", "The report is about 2025."),
    ("Rotate keys every 90 days.", "Keys are rotated every 90 days."),
]

DAMAGED: list[tuple[str, str, str]] = [
    # a source number turned into a date
    (KESTREL, "The Kestrel checkpoint was recalled at rank 1 on 2004-06-16.", "number_became_date"),
    (KESTREL, "The recall happened on June 1st, 2004", "number_became_date"),
    (KESTREL, "The recall occurred in 2004", "number_became_date"),
    ("Release v4.1.21 shipped.", "The release shipped on 2021-04-01.", "number_became_date"),
    ("The p95 was 2019.3 ms.", "The p95 was measured on 2019-03-01.", "number_became_date"),
    ("Index build took 2023.11 s.", "The index was built in November 2023.", "number_became_date"),
    ("Score improved to 2020.5 points.", "The score improved in May 2020.", "number_became_date"),
    ("Upgrade to 3.2.24 is required.", "The upgrade is required by 2024-03-02.",
     "number_became_date"),
    ("Latency was 2018.7 ms under load.", "Under load on 2018-07-12 latency spiked.",
     "number_became_date"),
    ("The ratio settled at 2012.4.", "The ratio settled in April 2012.", "number_became_date"),
    ("Build 2025.9 passed.", "The build passed on 2025-09-01.", "number_became_date"),
    ("The checkpoint took 2001.2 ms.", "The checkpoint was taken in 2001.", "number_became_date"),
    ("Memory peaked at 2016.8 MB.", "Memory peaked in August 2016.", "number_became_date"),
    ("Patch 1.4.22 is live.", "The patch went live on 2022-01-04.", "number_became_date"),
    ("Throughput was 2010.10 req/s.", "Throughput was recorded on 2010-10-01.",
     "number_became_date"),
    # a measurement, amount, version or id the source does not contain
    (KESTREL, "The Kestrel checkpoint was recalled at rank 1 in 2014.6 ms.", "unsupported_number"),
    ("Costs rose 45% to $12.50.", "Costs rose 54% to $12.50.", "unsupported_number"),
    ("The API listens on port 8080.", "The API listens on port 8081.", "unsupported_number"),
    ("Fix bug #412.", "Fix bug #421.", "unsupported_number"),
    ("The cluster uses 4.5 GB.", "The cluster uses 45 GB.", "unsupported_number"),
    ("Upgrade to v2.3.1.", "Upgrade to v2.3.2.", "unsupported_number"),
    ("Recall took 3.0 s.", "Recall took 30 s.", "unsupported_number"),
    ("We store 1,200 records per batch.", "Each batch holds 12,000 records.",
     "unsupported_number"),
    ("Error rate is 0.5%.", "Error rate is 5%.", "unsupported_number"),
    ("The plan costs $12.50.", "The plan costs $125.", "unsupported_number"),
    # a date the source never states
    ("The meeting moved to the library.", "On 2026-10-06 the meeting moved to the library.",
     "unsupported_date"),
    ("The launch is on 6 Oct 2026.", "The launch is on 7 October 2026.", "unsupported_date"),
    ("The audit found two gaps.", "The audit on 2026-09-30 found two gaps.", "unsupported_date"),
    ("Kickoff was March 15, 2026.", "Kickoff happened on 2026-03-16.", "unsupported_date"),
    ("The team moved to Lisbon.", "In 2023 the team moved to Lisbon.", "unsupported_date"),
    ("The contract ends 31/12/2026.", "The contract ends on 2026-11-30.", "unsupported_date"),
    ("The vendor signed the agreement.", "The vendor signed the agreement on June 3, 2026.",
     "unsupported_date"),
    ("Payment is due on 2026-04-30.", "Payment is due on 2026-05-30.", "unsupported_date"),
    ("The server was rebooted.", "The server was rebooted on 2026-10-01.", "unsupported_date"),
    ("The freeze starts in June 2026.", "The freeze starts in July 2026.", "unsupported_date"),
    # a dropped never / not / without
    (POLICY, "Publish with approval from the release owner.", "negation_lost"),
    ("We do not use Jenkins; we use GitHub Actions.", "The team uses Jenkins.", "negation_lost"),
    ("The cache is not enabled in production.", "The cache is enabled in production.",
     "negation_lost"),
    ("Never store tokens in the repository.", "Store tokens in the repository.", "negation_lost"),
    ("The job does not retry failed uploads.", "The job retries failed uploads.",
     "negation_lost"),
    ("Guests cannot edit the board.", "Guests can edit the board.", "negation_lost"),
    ("Do not deploy on Fridays.", "Deploy on Fridays.", "negation_lost"),
    ("The service doesn't support IPv6.", "The service supports IPv6.", "negation_lost"),
    ("No customer data leaves the region.", "Customer data leaves the region.", "negation_lost"),
    ("The report is not shared with vendors.", "The report is shared with vendors.",
     "negation_lost"),
    ("Never merge without a passing build.", "Merge with a passing build.", "negation_lost"),
    ("Alice will not attend the review.", "Alice will attend the review.", "negation_lost"),
    # a reversed order
    (ORDER, "Ship the Android build before the iOS build.", "order_reversed"),
    ("Ship iOS before Android.", "Ship Android before iOS.", "order_reversed"),
    ("Run the migration before the deploy.", "Run the deploy before the migration.",
     "order_reversed"),
    ("Back up the database before the upgrade.", "Upgrade before backing up the database.",
     "order_reversed"),
    ("Staging comes before production.", "Production comes before staging.", "order_reversed"),
    ("Review the design before coding.", "Start coding before the design review.",
     "order_reversed"),
    ("The audit happens after the release.", "The release happens after the audit.",
     "order_reversed"),
    ("Tests run before the build.", "The build runs before the tests.", "order_reversed"),
    ("Pay the invoice after the delivery.", "Pay the delivery after the invoice.",
     "order_reversed"),
    ("Migrate users before deleting the old table.",
     "Delete the old table before migrating users.", "order_reversed"),
    # "previously" lost: what was is stated as what is
    (STATUS, "The team uses Jenkins for builds.", "status_lost"),
    ("We formerly hosted on Heroku; now we host on Fly.", "We host on Heroku.", "status_lost"),
    ("The API used to return XML, but now it returns JSON.", "The API returns XML.",
     "status_lost"),
    ("Earlier the limit was 100, currently it is 500.", "The limit is 100.", "status_lost"),
    ("Previously Bob led the team, now Carol leads it.", "Bob leads the team.", "status_lost"),
    ("The office was previously in Pune, currently in Mumbai.", "The office is in Pune.",
     "status_lost"),
    ("Originally the app used MongoDB, now it uses Postgres.", "The app uses MongoDB.",
     "status_lost"),
]

REASONS = ("number_became_date", "unsupported_date", "unsupported_number",
           "negation_lost", "order_reversed", "status_lost")
