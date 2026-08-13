# Trend teacher asset

This directory contains the portable, verified Trend-teacher checkpoint used by
PropEvolve's training-only teacher pipeline.

- Checkpoint: `trend_teacher.pt`
- SHA-256: `63e24ad1ff661bcd7557063a928fc8400bec945e1097475e0b2b7a80ccb7fb8e`
- Adapter and cache implementation: `src/propevolve/teachers/trend.py`
- Authenticated metadata: `teachers/manifest.json` under the `trend` key
- Cache-build recipe: `config/trend_teacher_cache_v1.json`

The four soft channels are Long launch probability, Short launch probability,
Long conditional quality, and Short conditional quality. They are auxiliary
training targets, not observations, trading signals, or hard entry gates. The
teacher is removed before chronological selection.

Generate all nine local 3-minute, pre-2025 target caches before enabling a
Trend curriculum:

```bash
propevolve build-trend-teacher-cache \
  --config config/trend_teacher_cache_v1.json
```

Generated caches remain local and ignored by Git. The loader must fail closed
when a required ticker cache is missing or does not match the checkpoint,
frozen Mask representation, timestamps, or temporal boundary.
