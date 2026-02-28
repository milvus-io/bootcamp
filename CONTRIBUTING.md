# Contributing to Bootcamp

Thank you for your interest in contributing to the Milvus Bootcamp!

## Developer Certificate of Origin (DCO)

All commits in pull requests must be signed off to certify that you wrote
or otherwise have the right to submit the code under the project's license.
This is enforced automatically by the DCO check in CI.

To sign off a commit, use the `-s` (or `--signoff`) flag:

```bash
git commit -s -m "Your commit message"
```

This appends the following trailer to your commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

### Fixing missing sign-offs

If your pull request has commits that are missing a sign-off, you can
amend them with:

```bash
# Amend only the most recent commit
git commit --amend --signoff

# Sign off all commits in your branch at once (replace <base-branch> with
# the branch you opened the PR against, e.g. main)
git rebase --signoff <base-branch>
```

Then force-push the updated branch:

```bash
git push --force-with-lease
```

## Code Style

- Python files are formatted with [Black](https://black.readthedocs.io/).
  Run `black .` before opening a PR.
- Jupyter notebooks are also checked with Black's notebook support.

## Submitting a Pull Request

1. Fork the repository and create a new branch from `main`.
2. Make your changes and ensure all CI checks pass.
3. Open a pull request with a clear description of the changes.
