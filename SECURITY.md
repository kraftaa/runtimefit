# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's **Security → Report a
vulnerability** flow for this repository. Do not open a public issue containing a
credential, private benchmark payload, or exploitable detail.

Include the affected RuntimeFit version, reproduction steps, impact, and any proposed
mitigation. Maintainers will acknowledge the report and coordinate disclosure after a
fix is available.

## Sensitive benchmark data

RuntimeFit sanitizes recognized secret-bearing configuration keys before writing its
native result files, but users must still inspect configurations, workloads, provider
evidence, server logs, and generated reports before sharing them. Prefer environment
variable references over inline credentials and never publish private prompts.

Only run `runtimefit run --allow-processes` after reviewing every configured launch
command. That option executes local programs with the permissions of the current user.
