# Changelog

All notable project changes are documented here.

## [1.6.1] - 2026-07-29

### Added

- Average, P50, P90, P95, and P99 full-response latency statistics.
- Variable-controlled A100 reproduction of the upstream default core
  configuration.
- Official single-L20 streaming and offline reference results, with explicit
  workload and hardware comparability boundaries.

### Changed

- Benchmark system RTF now follows the upstream aggregate formula: profile
  wall time divided by total synthesized audio duration.
- README performance tables now report system RTF instead of average
  per-request RTF.
- JSON reports identify the metric standard and retain legacy per-request RTF
  fields only for compatibility and diagnostics.

## [1.6.0] - 2026-07-29

### Added

- GPU-memory-aware `balanced` and `throughput` performance profiles.
- Configurable BLS, token2wav, vocoder, Gateway, and long-text segment
  concurrency.
- `Server-Timing`, inference-time, and encode-time response headers.
- 16- and 24-concurrency A100 stress-test results.

### Changed

- 80 GB GPUs now use two token2wav and two vocoder instances while reducing
  oversized LLM KV-cache reservation.
- Throughput-profile Pro instances initialize CUDA before Triton reports ready,
  avoiding first-burst context creation.
- Long text can no longer consume every global inference slot from one
  request.
- Legacy BLS instances are reduced to reserve resources for the Public API.

## [1.5.1] - 2026-07-29

### Added

- Apache License 2.0, NOTICE, and Security Policy.
- Chinese and English launch-ready README pages.
- Real Web console workflow animation and social preview artwork.
- Reproducible Public API benchmark tool and A100 measurements.
- Structured Bug and Feature Issue forms, Pull Request template, and
  contribution guide.
- GitHub CI, unit tests, and tag-driven Release automation.

### Changed

- Repositioned the project around reusable Speaker Registry and
  production-ready CosyVoice serving.
- Updated GitHub Actions to Node 24 runtimes.

## [1.5.0] - 2026-07-29

### Added

- Same-port Web console and Public API Gateway on port `18000`.
- Public speaker registration, inspection, listing, and deletion APIs.
- Audio-file and public-URL registration.
- Unified `/tts/` endpoint with registered Speaker, built-in voice, and raw
  prompt-audio modes.
- Per-speaker default Prompt persona and per-request override.
- Server-side speed, volume, chunking, and audio-format post-processing.

[1.6.1]: https://github.com/QuadraV-Speech/CosyVoice3Pro/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/QuadraV-Speech/CosyVoice3Pro/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/QuadraV-Speech/CosyVoice3Pro/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/QuadraV-Speech/CosyVoice3Pro/releases/tag/v1.5.0
