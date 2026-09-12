from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtimefit import __version__
from runtimefit.config import ConfigError, load_config
from runtimefit.decision import choose
from runtimefit.decision_config import load_decision_config
from runtimefit.decision_report import decision_markdown, format_failed_checks
from runtimefit.environment import collect_environment
from runtimefit.report import markdown_report
from runtimefit.runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runtimefit",
        description="Turn LLM benchmark evidence and production constraints into a deployment decision.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="show benchmark-relevant system and tool metadata"
    )
    doctor.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    validate = commands.add_parser("validate", help="validate a decision configuration")
    validate.add_argument("config")
    validate_benchmark = commands.add_parser(
        "validate-benchmark", help="validate a built-in runner configuration"
    )
    validate_benchmark.add_argument("config")
    choose_command = commands.add_parser(
        "choose", help="select a deployment from benchmark evidence and requirements"
    )
    choose_command.add_argument(
        "config", help="decision config (.toml, .json, or optional .yaml)"
    )
    choose_command.add_argument(
        "--output", "-o", required=True, help="machine-readable decision JSON"
    )
    choose_command.add_argument("--report", help="optional Markdown decision report")
    run = commands.add_parser(
        "run", help="run the small built-in development evidence provider"
    )
    run.add_argument("config")
    run.add_argument("--output", "-o", required=True, help="result JSON path")
    run.add_argument("--report", help="optional Markdown report path")
    run.add_argument(
        "--allow-processes",
        action="store_true",
        help="allow reviewed launch.command entries to execute local runtime processes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            environment = collect_environment()
            if args.json:
                print(json.dumps(environment, indent=2))
            else:
                gib = (
                    environment["physical_memory_bytes"] / (1024**3)
                    if environment["physical_memory_bytes"]
                    else 0
                )
                print(f"RuntimeFit {environment['runtimefit']}")
                print(
                    f"System: {environment['platform']} ({environment['logical_cpu_count']} logical CPUs, {gib:.1f} GiB RAM)"
                )
                for name, details in environment["tools"].items():
                    status = (
                        details.get("version_output", "not found")
                        if details["available"]
                        else "not found"
                    )
                    print(f"{name}: {status}")
            return 0
        if args.command == "choose":
            config = load_decision_config(args.config)
            decision = choose(config)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
            if args.report:
                report = Path(args.report)
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(decision_markdown(decision), encoding="utf-8")
            summary = decision["summary"]
            print(
                f"{summary['eligible_count']}/{summary['candidate_count']} configurations satisfy requirements."
            )
            label = (
                "Provisional choice"
                if summary.get("selection_status") == "indistinguishable"
                else "Winner"
            )
            print(f"{label}: {summary['selected'] or 'none'}")
            if summary["reason"]:
                print(f"Reason: {summary['reason']}.")
            if summary.get("fastest_candidate") and not summary.get(
                "fastest_candidate_eligible"
            ):
                print(f"Fastest rejected: {summary['fastest_candidate']}")
                fastest = next(
                    candidate
                    for candidate in decision["candidates"]
                    if candidate["id"] == summary["fastest_candidate"]
                )
                for reason in format_failed_checks(fastest):
                    print(f"  - {reason}")
            print(f"Evidence: {output}")
            return 0 if summary["selected"] else 2
        if args.command == "validate":
            load_decision_config(args.config)
            print(f"Valid decision config: {args.config}")
            return 0
        config = load_config(args.config)
        if args.command == "validate-benchmark":
            print(f"Valid benchmark config: {args.config}")
            return 0
        result = run_benchmark(config, allow_processes=args.allow_processes)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.report:
            report = Path(args.report)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(markdown_report(result), encoding="utf-8")
        recommendation = result["recommendation"]["target"]
        print(f"Result: {output}")
        if args.report:
            print(f"Report: {args.report}")
        print(f"Recommendation: {recommendation or 'none'}")
        return 0 if recommendation else 2
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
