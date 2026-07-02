# Homebrew distribution via cargo-dist Implementation Plan

> **For agentic workers:** This plan is executed INLINE (config + CI + docs, no unit-testable logic). Steps use checkbox (`- [ ]`) syntax for tracking. Verification is tool-driven (`dist plan` / `dist build`), not TDD.

**Goal:** On every `git tag vX.Y.Z`, a GitHub Actions pipeline builds the Apple Silicon binary, publishes a GitHub Release, and pushes a Homebrew formula to `MaximeDeconinck/homebrew-tap`, so users install with `brew install maximedeconinck/tap/paddock` or `curl … | sh`.

**Architecture:** Use the `dist` CLI (cargo-dist by axodotdev). It reads config from the workspace, generates `.github/workflows/release.yml`, builds artifacts, cuts the GitHub Release, and publishes the Homebrew formula + shell installer. Two repos: `paddock` (source + releases), `homebrew-tap` (generated formula only).

**Tech Stack:** cargo-dist (`dist`), GitHub Actions, Homebrew, Rust 2024 workspace.

**Spec:** `docs/superpowers/specs/2026-07-02-homebrew-distribution-design.md`

**Locked config:** target `aarch64-apple-darwin` only · installers `homebrew` + `shell` · tap `MaximeDeconinck/homebrew-tap` · first tag `v0.1.0`.

---

## Prerequisites (user actions, tracked but not done by the implementer)

These are needed only for the LIVE end-to-end (the tag push after merge), not for the in-repo work or local verification:

- [ ] User creates empty public repo `MaximeDeconinck/homebrew-tap`.
- [ ] User generates a GitHub PAT with write/contents scope on the tap and stores it as a secret in the `paddock` repo. The exact secret name is whatever `dist init` reports it expects (see Task 2 Step 4); document it in the plan checkboxes once known.

The in-repo tasks below do NOT depend on these and can be completed + merged first.

---

## Task 1: Install `dist` and add the repository field

**Files:**
- Modify: `Cargo.toml` (`[workspace.package]`)

- [ ] **Step 1: Install the `dist` CLI**

Run: `brew install dist`
Expected: `dist` on PATH. Verify: `dist --version` prints a version (e.g. `dist 0.x.y`).

If `brew install dist` fails (formula name drift), fall back to: `curl --proto '=https' --tlsv1.2 -LsSf https://github.com/axodotdev/cargo-dist/releases/latest/download/cargo-dist-installer.sh | sh` then re-check `dist --version`.

- [ ] **Step 2: Add `repository` to `[workspace.package]`**

In `Cargo.toml`, the `[workspace.package]` table currently is:

```toml
[workspace.package]
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"
rust-version = "1.88"
```

Add the `repository` line:

```toml
[workspace.package]
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"
rust-version = "1.88"
repository = "https://github.com/MaximeDeconinck/paddock"
```

- [ ] **Step 3: Verify the workspace still builds**

Run: `cargo build --workspace`
Expected: builds clean (adding `repository` is metadata-only).

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml
git commit -m "chore: add repository metadata for cargo-dist"
```

---

## Task 2: Initialize cargo-dist config + generate CI

**Files:**
- Create: `dist-workspace.toml` (or `[workspace.metadata.dist]` in `Cargo.toml` - dist's choice by version)
- Create: `.github/workflows/release.yml`
- Possibly modify: `Cargo.toml` (dist may add a `[workspace.metadata.dist]` block and/or a profile)

- [ ] **Step 1: Run `dist init`**

Run: `dist init`

`dist init` is interactive. Answer the prompts to match the locked config:
- **Targets:** select ONLY `aarch64-apple-darwin` (Apple Silicon macOS). Deselect all others.
- **Installers:** enable `homebrew` and `shell`; leave the rest off.
- **Homebrew tap:** when asked for the tap repo, enter `MaximeDeconinck/homebrew-tap`.
- **CI:** GitHub Actions (the only/ default backend).

If a non-interactive run is preferred, `dist init --yes` accepts defaults; then hand-edit the generated config (Step 2) to the locked values and re-run `dist generate`. Either path is fine as long as the final config matches the locked values.

- [ ] **Step 2: Verify the generated config matches the locked values**

Open the config `dist init` wrote (`dist-workspace.toml` at the workspace root on recent `dist`, or a `[workspace.metadata.dist]` table in `Cargo.toml` on older versions). Confirm it contains:
- targets = exactly `["aarch64-apple-darwin"]`
- installers include `"homebrew"` and `"shell"` and nothing else
- a homebrew tap set to `"MaximeDeconinck/homebrew-tap"`

If any value is wrong, edit the file directly to the locked value, then run `dist generate` (Step 3) to propagate to the workflow.

- [ ] **Step 3: (Re)generate the CI workflow**

Run: `dist generate`
Expected: `.github/workflows/release.yml` exists (created or refreshed) with no diff drift complaints. `dist generate` is idempotent; running it after a manual config edit syncs the workflow.

- [ ] **Step 4: Note the tap-token secret name**

Read the generated `.github/workflows/release.yml` and find where the Homebrew publish step references a secret (a `secrets.<NAME>` for pushing to the tap). Record that exact `<NAME>` here and in the Prerequisites checklist so the user creates the matching secret:

Secret name expected by the workflow: `__________` (fill in from the generated file).

- [ ] **Step 5: Commit the config + workflow**

```bash
git add dist-workspace.toml Cargo.toml .github/workflows/release.yml
git commit -m "ci: cargo-dist release pipeline (aarch64-apple-darwin, homebrew + shell)"
```

(If `dist` used `[workspace.metadata.dist]` in `Cargo.toml` instead of a separate `dist-workspace.toml`, adjust the `git add` to the files that actually changed. Include `Cargo.lock` if `dist init` touched it.)

---

## Task 3: Verify the release plan and a local build

**Files:** none (verification only)

- [ ] **Step 1: `dist plan`**

Run: `dist plan`
Expected: succeeds and lists a single `aarch64-apple-darwin` artifact for the `paddock` binary, plus a `homebrew` installer and a `shell` installer. No target other than `aarch64-apple-darwin`. `paddock-core` must NOT appear as a distributed binary.

If `dist plan` reports config errors, fix the config (Task 2 Step 2) and re-run.

- [ ] **Step 2: `dist build` locally**

Run: `dist build`
Expected: produces the arm64 binary archive under `target/distrib/` (or the path `dist` reports) AND a generated Homebrew formula (`paddock.rb`) that references the artifact URL and a `sha256`. Confirms the macOS GUI deps (`tray-icon`/`tao`/`arboard`) compile for the native target.

If the build fails on the GUI deps, STOP and report - do not proceed to tagging. This is the risk called out in the spec.

- [ ] **Step 3: Inspect the generated formula**

Read the generated `paddock.rb` (path from `dist build` output). Confirm:
- it installs the `paddock` binary,
- its `url` points at a `github.com/MaximeDeconinck/paddock/releases/...` artifact,
- it carries a `sha256`.

No commit in this task (build outputs are not checked in; verify `target/` is gitignored - it is, in a standard Rust repo).

---

## Task 4: Update the README Install section

**Files:**
- Modify: `README.md` (the `## Install` section, ~lines 23-30)

- [ ] **Step 1: Read the current Install section**

Run: `sed -n '20,35p' README.md` to see the exact current text (the `cargo install --path crates/paddock` block and the "A Homebrew formula is coming. Requires macOS on Apple Silicon (M1 or later)." line).

- [ ] **Step 2: Rewrite the Install section**

Replace the section so it reads (adapt surrounding markdown/headers to the file's existing style):

```markdown
## Install

Homebrew (recommended):

```
brew install maximedeconinck/tap/paddock
```

Or the install script:

```
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/MaximeDeconinck/paddock/releases/latest/download/paddock-installer.sh | sh
```

From source (needs a Rust toolchain):

```
cargo install --path crates/paddock
```

Requires macOS on Apple Silicon (M1 or later).
```

NOTE: confirm the exact installer script filename from Task 3's `dist plan`/`dist build` output (cargo-dist names it `<app>-installer.sh`; for this workspace that is `paddock-installer.sh`, but verify against the tool output and correct the URL if it differs).

Use `-` or ` · ` as separators. The em-dash `—` is BANNED project-wide.

- [ ] **Step 3: Verify no em-dash**

Run: `grep -n "—" README.md`
Expected: no output (exit 1). If a pre-existing em-dash exists in an untouched section, leave it; do not ADD any.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: install via Homebrew tap and install script"
```

---

## Task 5: Finalize the branch

**Files:** none

- [ ] **Step 1: Full build + test sanity**

Run: `cargo test --workspace`
Expected: green (this change is config/docs; no code behavior changed).

- [ ] **Step 2: Confirm the branch diff is clean and scoped**

Run: `git diff main --stat`
Expected: only `Cargo.toml`, `dist-workspace.toml` (or the metadata block), `.github/workflows/release.yml`, `README.md`, the spec/plan docs, and possibly `Cargo.lock`. No stray source changes.

- [ ] **Step 3: Hand off to finishing-a-development-branch**

Use the `superpowers:finishing-a-development-branch` skill to push and open a PR. The PR description must include the post-merge user runbook:
1. create `MaximeDeconinck/homebrew-tap` (empty, public),
2. add the PAT secret (name from Task 2 Step 4) to the `paddock` repo,
3. `git tag v0.1.0 && git push origin v0.1.0`,
4. confirm the release CI is green, the tap gets `Formula/paddock.rb`, and `brew install maximedeconinck/tap/paddock` works.

---

## Self-Review notes

- **Spec coverage:** Task 1 (repository metadata + tool) · Task 2 (dist config + workflow + tap + token-secret name) · Task 3 (dist plan/build verification, GUI-deps risk gate) · Task 4 (README) · Task 5 (test sanity + PR with the user runbook). All spec components mapped.
- **No TDD:** deliberate - packaging has no unit-testable units (spec's Execution note). Verification is `dist plan`/`dist build`.
- **Interactive `dist init`:** flagged in Task 2 Step 1 with a non-interactive fallback; the final-config check (Step 2) is the real gate regardless of path.
- **Unknowns resolved at execution time, not left as placeholders:** the tap-token secret name (Task 2 Step 4) and the exact installer-script filename (Task 4 Step 2) are read from the tool's own output during execution - they are tool-authoritative values, not guesses to hardcode.
- **User-owned steps** (tap repo, PAT secret, tag push) are isolated in Prerequisites + Task 5 Step 3 so the in-repo work merges independently.
