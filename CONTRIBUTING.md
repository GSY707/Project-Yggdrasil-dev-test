# Contributing to Project Yggdrasil

Thanks for your interest in contributing.

## Ground Rules

- Be respectful and constructive in discussions.
- Keep changes focused and atomic.
- Include tests for behavior changes when possible.
- Do not commit secrets, tokens, or local databases.

## Development Setup

1. Fork and clone the repository.
2. Create a feature branch from `main`.
3. Install dependencies in your Python environment.
4. Run tests before opening a pull request.

## Commit and Pull Request Guidelines

- Write clear commit messages.
- Link related issues in the pull request description.
- Describe what changed, why, and how it was tested.
- Keep pull requests reviewable in size.

## Testing

Run project tests from the `src/` directory:

`python -m pytest`

If your change affects async web behavior, include manual verification notes.

## Code Style

- Preserve existing module structure and naming conventions.
- Prefer explicit, readable code over clever shortcuts.
- Add concise comments only where logic is non-obvious.

## Licensing

By contributing, you agree that your contributions are licensed under AGPL-3.0.
