# Contributing

## Branches

Create a branch from `main`:

```bash
git checkout -b feature/short-description
```

Use branch prefixes:

- `feature/` for new functionality
- `fix/` for bug fixes
- `docs/` for documentation
- `hardware/` for hardware design changes
- `firmware/` for embedded changes

## Pull Requests

Before opening a pull request:

- Keep changes focused.
- Add or update tests when behavior changes.
- Update docs for new setup steps, interfaces, or design decisions.
- Mention related issues.
- Request at least one teammate review.

## Commit Messages

Use clear, action-oriented messages:

```text
Add BLE packet parser
Document sensor calibration flow
Fix dashboard connection status
```

## Code And Design Reviews

Reviewers should check:

- Correctness and edge cases
- Test coverage
- Hardware/software interface assumptions
- Documentation updates
- Maintainability for teammates

## File Ownership

Coordinate before editing files that define shared contracts, such as API schemas, packet formats, pin mappings, and data schemas.
