# Releasing RuntimeFit

RuntimeFit has two distribution channels:

- PyPI provides `pip install runtimefit`.
- A personal Homebrew tap provides `brew install kraftaa/tap/runtimefit`.

Do not publish until the `runtimefit` name, repository owner, metadata, and version are
final. PyPI releases are effectively permanent; versions cannot be reused.

## One-time PyPI setup

1. Push this project to `kraftaa/runtimefit` on GitHub.
2. Create a PyPI account with two-factor authentication.
3. On PyPI's pending publisher page, register:
   - project: `runtimefit`
   - owner: `kraftaa`
   - repository: `runtimefit`
   - workflow: `release.yml`
   - environment: `pypi`
4. In GitHub, create an environment named `pypi` and require manual approval.

No PyPI API token should be added to GitHub. The workflow uses short-lived OIDC
credentials and publishes attestations through PyPA's publishing action.

## One-time Homebrew setup

Create a separate public repository named `kraftaa/homebrew-tap`. Its layout should be:

```text
homebrew-tap/
└── Formula/
    └── runtimefit.rb
```

After every tagged RuntimeFit build, the release workflow produces a
`homebrew-formula` artifact. Copy its `runtimefit.rb` into the tap's `Formula/`
directory, review it, test it, and commit it.

Users can then install RuntimeFit with:

```bash
brew install kraftaa/tap/runtimefit
```

## Publishing a version

1. Update `project.version` in `pyproject.toml`.
2. Run tests and build checks locally.
3. Commit the release changes.
4. Create and push the matching tag, such as `v0.1.0`.
5. Approve the protected `pypi` environment deployment.
6. Download and test the generated Homebrew formula artifact.
7. Commit the formula to `kraftaa/homebrew-tap`.

The workflow rejects a tag when its version does not exactly match `pyproject.toml`.

## Local formula generation

After building an sdist:

```bash
python scripts/render_homebrew_formula.py \
  --version 0.1.0 \
  --sdist dist/runtimefit-0.1.0.tar.gz \
  --github-owner kraftaa \
  --output /path/to/homebrew-tap/Formula/runtimefit.rb
```

Validate inside the tap using Homebrew's audit and test commands before publishing.
