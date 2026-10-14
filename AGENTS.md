# Repository Guidelines

## Project Structure & Module Organization

This workspace currently contains no source code, tests, assets, or package manifests. The `.agents/`, `.codex/`, `.aws/`, and `.git/` directories are reserved for environment metadata; do not use them for application code.

When introducing the initial implementation, keep the layout explicit: use `src/` for source modules, `tests/` for automated tests, and `assets/` only if static resources are needed. Add a `README.md` describing the project's purpose, setup, and entry points. These paths are proposed conventions, not existing directories.

## Build, Test, and Development Commands

No build, test, or local development commands are configured yet. Do not assume commands such as `npm test`, `pytest`, or `make build` work in this workspace.

When adding a toolchain, document exact installation, development, build, and test commands in `README.md`. Commit the appropriate dependency manifest and lockfile, and keep documented commands consistent with the configuration.

## Coding Style & Naming Conventions

No language, formatter, or linter has been selected. Follow the chosen language's standard conventions and configure formatting and linting when introducing source code. Use consistent indentation within each language and avoid unrelated formatting changes.

Choose descriptive module, function, and variable names. Organize modules by responsibility, keep public interfaces small, and document behavior that is not apparent from the code.

## Testing Guidelines

No testing framework or coverage threshold exists yet. Add automated tests with the first executable functionality and document how to run them. Use descriptive test names that identify the behavior and expected result; mirror source module organization under `tests/` where the framework supports it.

Cover normal behavior, boundary cases, and relevant failure paths. Run the applicable tests before submitting changes and report any checks that could not be run.

## Commit & Pull Request Guidelines

Git history is unavailable in this workspace, so no existing commit convention can be inferred. Use short, imperative commit subjects, such as `Add initial project setup`, and keep commits focused.

Pull requests should explain the change, its purpose, and validation performed. Link related issues when available. Include screenshots for visual changes and document new dependencies, configuration requirements, or breaking changes.

## Security & Configuration

Keep credentials, tokens, and local configuration out of committed files. Document required environment variables using placeholder values, and add ignore rules for generated outputs and local secrets when initializing the project.
