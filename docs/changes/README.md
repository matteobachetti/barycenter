# Changelog fragments

One file per user-visible change, named `<PR number>.<type>.md`, collected into
`CHANGELOG.md` by [towncrier](https://towncrier.readthedocs.io) at release time.

Get the number for the next pull request with
[`nextpr`](https://github.com/matteobachetti/nextpr):

```bash
nextpr
```

Types: `breaking`, `feature`, `bugfix`, `doc`, `maintenance`.

```bash
echo "Read Chandra orbit files natively, so DE440 is reachable." > docs/changes/42.feature.md
```

Write the fragment for the person reading the release notes: what changed for them,
in at most two sentences, not what the diff did.
