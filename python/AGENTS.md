# Repository instructions

- Put every new Python test module or test-only executable under `test/`.
- Keep production modules, reusable test helpers, fixtures, and test data in their owning packages when appropriate, but do not add `test_*.py`, `*_test.py`, or other test-only scripts outside `test/`.

## Long-running experiment follow-through

- User preference saved on 2026-10-16: when the user asks to launch training experiments and perform diagnostics or follow-up trials after completion, they authorize temporary continuations in the same chat to carry out that requested sequence. Do not ask again for continuation permission within this scope.
- Prefer a temporary thread heartbeat, check for an existing matching continuation before creating one, and pause it once the requested study and report are complete. Keep updates quiet unless there is a meaningful result, failure, or required user action.
- This preference covers the requested experiment sequence; it does not authorize unrelated experiments, an unbounded parameter search, or changes to other running jobs.
- User preference on 2026-10-17 for the full-dataset threshold-10 rerun: after verifying launch, stop periodic checking. Do not create a heartbeat for this sequence. Use the local sequential controller to wait for child exits and advance the already authorized scene queue; respond to later explicit status requests normally.

