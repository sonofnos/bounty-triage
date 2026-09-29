# Triage playbook

How a security report moves from inbox to disclosure, and where the model helps. The model drafts; a person decides every step.

## 1. Acknowledge (target: 1 business day)

- Reply to the reporter, confirm receipt, give a reference number and the next update date.
- Run `python -m bounty_triage triage report.md` and attach the note to the ticket.
- If the note says **likely duplicate**, read the matched advisory before anything else. A duplicate is closed only after a person confirms it is the same root cause, not the same component. Evaluation shows the model's most confident mistakes are different bugs in the same crate.

## 2. Reproduce

- Reproduce on the version the reporter named, then on the latest release and `main`. Record commit hashes.
- For runtime (pallet) reports: reproduce in a mock runtime test first, then on a local dev chain. A report is not confirmed until there is a failing test or a transaction trace.
- If it cannot be reproduced, ask the reporter for the exact steps before downgrading. Say what was tried.

## 3. Assess impact and severity

Set severity from demonstrated impact, never from the reporter's claim or the model's suggestion. For blockchain infrastructure the questions that matter:

| Impact | Typical severity |
|---|---|
| Loss or theft of user funds; minting; breaking total issuance | Critical |
| Consensus failure, chain halt, or a block that cannot be imported | Critical |
| Permanent freezing of funds; bypass of governance or origin checks | High |
| Free or underpriced block space (mis-weighted calls), state bloat at fixed cost, panics a caller can trigger without paying | Medium to High, depending on cost to attack |
| Node crash or resource exhaustion through RPC or networking | Medium to High, depending on exposure |
| Information leaks, incorrect events, UX-level issues | Low |

Record a CVSS v3.1 vector alongside the programme severity. `bounty_triage.cvss.base_score` computes the base score from a vector. Where they disagree, the programme's impact table wins and the reason is written down.

## 4. Assign and fix

- Owner is the team that owns the code, with a security engineer as reviewer.
- The fix lands with a regression test that fails before the fix. For pallets: a unit test, plus an invariant in `try_state` where the bug broke one.
- Look for the same bug class elsewhere: a static-analysis rule, a grep, or a fuzz target. One report often points at several instances.

## 5. Disclose and reward

- Agree the disclosure date with the reporter. Ship the fix, then publish the advisory with credit.
- Pay according to confirmed severity. If severity changed during assessment, tell the reporter why.
- Add the advisory to the index so the next duplicate is caught at step 1.

## What the model does, and what it does not

| Model output | How to use it |
|---|---|
| Closest known advisories | Read them. They are the fastest way to spot a duplicate or a known bug class. |
| Likely / possible duplicate | A prompt to compare, never an auto-close. |
| Vulnerability class | Routing: which reviewer to pull in first. |
| Severity suggestion | Barely better than always guessing "high" (see `reports/RESULTS.md`). Treat it as a placeholder until step 3. |
