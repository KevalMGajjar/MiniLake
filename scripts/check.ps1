# CI-style checks: formatting, lints (warnings are errors), tests.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
cargo fmt --all -- --check
if (-not $?) { exit 1 }
cargo clippy --workspace --all-targets -- -D warnings
if (-not $?) { exit 1 }
cargo test --workspace
if (-not $?) { exit 1 }
