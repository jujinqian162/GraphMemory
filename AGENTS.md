# Repository Agent Instructions

## PowerShell tools

The following enhanced PowerShell tools are available:

- `rg` for content search. With the profile enabled, `rgp` maps to
  `rg.exe --path-separator="/" @args` for easier `xargs` integration.
- `eza` for directory listings, including `eza -T`.
- `bat` for file output with colors and line numbers.
- `qsv` for CSV processing.
- `jq` and `yq` for JSON and YAML processing.
- `xargs` when the PowerShell profile is enabled.
- `fd` for file search.
- `ast-grep` for AST-based search.
- `ck --sem "prompt"` for semantic search.
- `procs` for process listings.

Available environment managers include `uv` and `bun`.

## Windows sandbox execution

On this Windows host, run every `uv` command outside the Codex filesystem
sandbox from the first attempt. Use `sandbox_permissions="require_escalated"`
and request a narrow reusable prefix when possible, such as:

- `["uv", "run", "pytest"]`
- `["uv", "run", "ruff"]`
- `["uv", "run", "basedpyright"]`

Do not work around `uv` cache failures by creating a repository-local uv cache.
Outside the sandbox, `uv` must use the same user cache and environment as the
interactive PowerShell session.

Never run pytest inside the Codex Windows sandbox when tests can create
temporary directories. Pytest creates those directories with mode `0o700`,
which becomes an unusable Windows ACL under the restricted sandbox token and
can leave directories that the normal user cannot delete. In particular:

- Do not create `.pytest_tmp`, `.pytest-tmp`, or repo-local `--basetemp` paths.
- Do not treat a different `--basetemp` location as an ACL fix.
- Run the normal command, such as `uv run pytest ...`, outside the sandbox.

The repository `.codex/config.toml` requests `danger-full-access`, but a
session-level permission profile may override it. Follow the rules above based
on the effective tool permissions, not only that file.

## Windows apply_patch workaround

On this Windows host, the `apply_patch` entrypoint can fail before patch
parsing because the generated `apply_patch.bat` points at the WindowsApps
`codex.exe`, which may be blocked by the sandbox with `Access is denied` /
`CreateProcessAsUserW failed: 5`.

When `apply_patch` fails with a Windows sandbox wrapper error, do not keep
retrying the same tool path. Prefer one of these fallbacks:

- Invoke the copied Codex binary directly:

  ```powershell
  $patch = @'
  *** Begin Patch
  *** Update File: path/to/file
  @@
  -old
  +new
  *** End Patch
  '@

  & "$HOME\.codex\.sandbox-bin\codex.exe" --codex-run-as-apply-patch $patch
  ```

- For large documents or broad mechanical edits, use small, targeted
  PowerShell file rewrites or replacements, then immediately verify with
  `git diff`, `rg`, or the relevant targeted test.

Keep patch payloads small on Windows. Avoid giant monolithic patch strings:
they can hit wrapper, sandbox, or command-length failures and leave partial
files behind.
