# Security Policy

## Supported versions

Security fixes are applied to the latest `1.5.x` release line and `master`.
Older snapshots may not receive backports.

## Reporting a vulnerability

Please do not publish credentials, private audio, Speaker Registry data, or
full exploit details in a public Issue.

Use GitHub's private vulnerability reporting flow from the repository
**Security** tab when it is available. If private reporting is unavailable,
open a minimal Issue titled `[Security] Contact request` without sensitive
details so a maintainer can establish a private channel.

Include the affected version, deployment topology, reproduction conditions,
impact, and any proposed mitigation.

## Deployment boundary

CosyVoice3Pro does not enable application-level authentication by default.
Internet-facing deployments must add TLS, authentication and authorization,
request-size and rate limits, source-network restrictions, and independent
Speaker Registry backups at the reverse proxy or load balancer.
