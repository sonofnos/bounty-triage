# Triage: Rojo's "rojo serve" HTTP API (default port 34872) has no Host/Origin header validation, making it vu

**Likely duplicate of RUSTSEC-2026-0279** (rojo; duplicate probability 0.88). Confirm and close as known, or explain what is new.

## Suggested class

- format-injection: 0.51
- code-execution: 0.44  <-
- file-disclosure: 0.38
- memory-corruption: 0.09

## Suggested severity

high (0.64). From text alone; set the real value from a CVSS vector or the programme's impact table after reproducing.

## Closest known advisories

| advisory | crate | similarity | title |
|---|---|---|---|
| RUSTSEC-2026-0279 | rojo | 0.45 | Rojo development server vulnerable to DNS rebinding, allowing unauthen |
| RUSTSEC-2026-0140 | dynoxide-rs | 0.17 | DNS rebinding and cross-origin CSRF in dynoxide's MCP HTTP transport |
| RUSTSEC-2026-0189 | rmcp | 0.16 | DNS rebinding vulnerability in rmcp Streamable HTTP server transport |
| RUSTSEC-2025-0146 | sha-rust | 0.09 | `sha-rust` was removed from crates.io for malicious code |
| RUSTSEC-2026-0035 | pingora-cache | 0.09 | Cache poisoning via insecure-by-default cache key |

## Next steps

- [ ] Reproduce on the affected version; record exact commit and steps
- [ ] Confirm impact and severity; agree it with the reporter
- [ ] Assign an owner and a fix deadline from the severity SLA
- [ ] Coordinate disclosure date; credit the reporter
