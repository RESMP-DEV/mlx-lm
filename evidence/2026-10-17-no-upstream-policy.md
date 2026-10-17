# No-upstream standing policy

Date: 2026-10-17

## Scope

The user set a permanent boundary: MLX-LM work is never contributed upstream to
`ml-explore/mlx-lm`. The maintainers do not want our contributions and have not
asked for them. All our work is published on RESMP-DEV instead.

## Changes

- `AGENTS.md`: replaced inherited upstream-PR ban language with a standing
  no-upstream policy. RESMP-DEV pushes and pull requests remain allowed.
- The checkout is named as the RESMP-DEV research fork, matching its public
  origin.

## Remote enforcement

The policy is permanent for `ml-explore/mlx-lm`. The organization-owned
policy and audit receipts live in private `RESMP-DEV/agent-policy`.


The upstream push URL points to the deliberately invalid sentinel
`/Users/kearm/.config/no-upstream-push`; the fetch URL remains unchanged. A
dry-run push to `upstream` fails immediately with exit 128 and never reaches
the network. `git ls-remote upstream refs/heads/main` still returns the current
upstream head.

## Non-claims

This receipt does not imply an upstream reviewer opinion beyond the recorded
boundary, does not modify any upstream repository, and does not withdraw prior
publication from the public RESMP-DEV remote.

## Machine-wide enforcement

The same audit found eleven other local checkouts with writable upstream push
URLs. All thirteen local repositories with an `upstream` remote now share the
`~/.config/no-upstream-push` sentinel, verified by a failing dry-run push in
each. The canonical rule lives in the machine-wide `~/AGENTS.md` core as a
permanent no-upstream policy; the machine-wide audit receipt is
`evidence/2026-10-17-machine-wide-upstream-audit.md` in `/Users/kearm/mlx`.
