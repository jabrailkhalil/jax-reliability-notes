# Testing numerical APIs that return plausible wrong answers

Prepared 10-minute technical session by Jabrail Khalil. Both examples use CPU; no accelerator is required. This is an outline, not a record of a delivered session.

## 0:00–1:00 — Establish the testing loop

- Write an independent reference result before looking at the implementation.
- Measure both the forward value and gradient.
- Find the smallest responsible wrapper or optimization.
- Add a failing regression and retain neighboring checks.

## 1:00–4:00 — A keyword changes the intended loss

1. Show `sum(x * scale)` with three ones and scale 0.5.
2. Ask the audience to calculate the expected value and derivative.
3. Run `python examples/keyword_forwarding.py --expect baseline`.
4. Follow the keyword dictionary into `argnums_partial2`.
5. Apply the keyword patch and rerun with `--expect fixed`.
6. Explain why required keywords, auxiliary output, and changed jitted inputs need coverage.

## 4:00–8:00 — A group-size shortcut changes the mean

1. Draw device groups `[0]` and `[1, 2, 3]`.
2. Calculate the expected per-group mean and `psum(1)`.
3. Run `python examples/uneven_collectives.py --expect baseline`.
4. Explain when multiplication by one shared size is valid.
5. Apply the collective patch and rerun with `--expect fixed`.
6. Derive the gradient of the sum of replicated means.
7. Explain the integer divisor and strict dtype-promotion check.

## 8:00–10:00 — Review and limitations

- Show the upstream regression-test changes and CPU evidence.
- State that the submitted PRs are awaiting review; a local passing suite is not a merge.
- Explain that GPU/TPU CI and the earlier collective revert need upstream review.
- Leave the audience with the pinned examples and their own independent reference calculation.

## Before delivering

- Follow the README setup in a fresh environment and run both stages.
- Prepare a second checkout with both patches applied so the demo can continue if the network is unavailable.
- Recheck PR status and update the opening statement if either contribution has merged or changed.
- After a real session, record its public event link and materials; add attendance only if the organizer supplies a reliable number.
