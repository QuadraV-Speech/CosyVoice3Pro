# Changelog

All notable project changes are documented here.

## [1.9.0] - 2026-08-05

### Added

- Official-compatible streaming benchmark using `seed_tts_cosy2`, one
  persistent gRPC stream per task, raw prompt audio, upstream input padding,
  upstream TTFA boundary, and per-stage Triton statistics.
- A measured `streaming` production profile for 80 GB GPUs and configurable
  first-chunk token count.
- SSE admission queue timeout with a structured `STREAM_BUSY` error event.

### Changed

- The 80 GB `auto` profile now prioritizes streaming with two Streaming BLS,
  two Flow, and four Vocoder instances; offline-heavy deployments can select
  `throughput` explicitly.
- FFmpeg starts only after Triton emits its first audio block, avoiding a
  subprocess storm while requests are waiting for TTFA.
- Disconnect cleanup releases capacity and terminates post-processing before
  waiting for a slow gRPC generator.
- Client cancellation now propagates into the Decoupled BLS loop so queued
  Flow/Vocoder work stops at the next stage boundary.
- The production A100 SSE soak completed 100/100 requests at concurrency 16
  with 17.05x audio throughput and 2.28 s TTFA P95.

## [1.8.0] - 2026-08-05

### Added

- Official-aligned streaming benchmark for direct Triton gRPC and Public SSE,
  including TTFA percentiles, system RTF, throughput, failures, and Gateway
  queue time.
- SSE `queue` event plus Triton-first-audio and post-processing timing fields.

### Changed

- The Gateway now reuses one asynchronous Triton gRPC channel while retaining
  an independent response iterator and cancellation path per request.
- Streaming Flow and Vocoder child requests use first-chunk priority to improve
  latency fairness under concurrent load.
- The A100 throughput profile accepts up to 10 Public SSE streams while keeping
  the empirically faster two Streaming BLS and two Vocoder instances.
- The throughput profile keeps the first acoustic chunk unchanged and grows
  later chunks more aggressively, reducing repeated Flow/Vocoder work.
- FFmpeg raw-input probing is minimized so processed PCM is emitted while the
  Triton stream remains open instead of being buffered until near completion.
- The Web console displays queue time separately from first-audio and total
  latency.
- In the controlled A100 streaming test, concurrency-16 audio throughput rises
  from 12.94x to 17.21x while system RTF falls from 0.0773 to 0.0581.

## [1.7.0] - 2026-08-05

### Added

- Public `POST /tts/stream` SSE interface with `meta`, incremental `audio`,
  `done`, and in-stream `error` events.
- Dedicated `CosyVoice3ProStreaming` Triton Decoupled model with configurable
  instances and Gateway concurrency.
- Web-console online playback that decodes PCM chunks as they arrive, supports
  cancellation, and provides a completed WAV download.
- Streaming speed, volume, resampling, disconnect cancellation, keepalive, and
  first-audio timing.

### Changed

- Aggregate health now includes the streaming model readiness state.
- Public, advanced, operations, Chinese, and English documentation now cover
  the online SSE path and runnable curl examples.

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

[1.9.0]: https://github.com/QuadraV-Speech/CosyVoice3Pro/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/QuadraV-Speech/CosyVoice3Pro/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/QuadraV-Speech/CosyVoice3Pro/compare/v1.6.1...v1.7.0
[1.6.1]: https://github.com/QuadraV-Speech/CosyVoice3Pro/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/QuadraV-Speech/CosyVoice3Pro/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/QuadraV-Speech/CosyVoice3Pro/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/QuadraV-Speech/CosyVoice3Pro/releases/tag/v1.5.0
