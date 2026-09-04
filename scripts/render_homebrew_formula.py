#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def render_formula(version: str, sha256: str, github_owner: str) -> str:
    return f'''# typed: strict
# frozen_string_literal: true

# RuntimeFit turns LLM benchmark evidence into deployment decisions.
class Runtimefit < Formula
  include Language::Python::Virtualenv

  desc "Make evidence-backed LLM inference deployment decisions"
  homepage "https://github.com/{github_owner}/runtimefit"
  url "https://github.com/{github_owner}/runtimefit/releases/download/v{version}/runtimefit-{version}.tar.gz"
  sha256 "{sha256}"
  license "MIT"

  depends_on "python@3.14"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "runtimefit {version}", shell_output("#{{bin}}/runtimefit --version")
  end
end
'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render RuntimeFit's Homebrew formula")
    parser.add_argument("--version", required=True)
    parser.add_argument("--sdist", required=True, type=Path)
    parser.add_argument("--github-owner", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    digest = hashlib.sha256(args.sdist.read_bytes()).hexdigest()
    formula = render_formula(args.version, digest, args.github_owner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(formula, encoding="utf-8")
    print(f"Wrote {args.output} (sha256: {digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
