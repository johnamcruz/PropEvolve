# Config-Driven Codebase Audit

Status: deferred until the active R19 campaign finishes.

This is a read-only audit of settings embedded in Python that can make a new
training recipe, replay source, checkpoint, curriculum, or launch require a
source-code change. R19 must not be modified or restarted for this cleanup.

## Governing rule

- Python implements reusable behavior and validates relationships.
- JSON or CLI selects run-specific values and artifacts.
- Python must not name a campaign, run, checkpoint instance, replay instance,
  episode, ticker universe, or machine-specific path.
- Stable serialization schemas, action enums, tensor shapes, and authenticated
  artifact compatibility rules remain code contracts.
- The normalized effective recipe must be persisted and must contain every
  value that affects training, validation, or launch behavior.

## Confirmed findings

### P0: entry supervision is configurable in appearance only

`src/propevolve/config.py` accepts an `entry_supervision` object, but validates
exact literals for five decisions, fill offsets, launch and continuation
thresholds, $300 risk, +2R/-1R economics, a 150-bar horizon, and the 0.8 teacher
autonomy boundary. `src/propevolve/entry_supervision.py` repeats the same values
in `_FIXED_SPEC` and returns that fixed object instead of the supplied recipe.

Impact: changing a valid economic experiment in JSON requires synchronized
Python edits in two modules. This is the clearest violation of config-driven
training.

Required correction:

1. Parse one typed entry-supervision recipe from JSON.
2. Validate relational invariants instead of today’s literal values:
   positive risk; target and stop positive; positive horizon; ordered unique
   fill offsets; phase horizons positive; and risk consistent with challenge
   risk.
3. Preserve next-bar-open execution and leakage-safe label construction as
   code invariants.
4. Pass the parsed recipe to the label builder without replacing its values.
5. Add TDD cases for the current recipe, one alternate valid recipe, and
   invalid relational recipes.

### P0: the legacy recovery curriculum hardcodes one experiment

`src/propevolve/config.py` and `RecoveryCurriculumSettings` in
`src/propevolve/training.py` require exactly -$2,000 realized/equity/session
PnL, a -$3,000 floor, $500 minimum headroom, $300 risk, and a $0 recovery
target.

Impact: changing the recovery starting balance or safety runway can require a
Python change even though those fields exist in JSON. This is the same failure
mode encountered during recovery iteration.

Required correction: validate start-state consistency and challenge-relative
relationships, not one historical Stage 2B recipe. A configured start must be
above its configured MLL floor; realized, equity, and session state must agree;
the configured recovery target must be reachable; and the safety threshold must
fit within configured MLL headroom.

### P1: normalized headroom strata are duplicated literals

The 0.25/0.75 headroom split is repeated in:

- `src/propevolve/agent.py`;
- `src/propevolve/training.py`;
- `src/propevolve/final_regime_probe.py`.

Dollar strata of $300/$500 are separately embedded in validation economics in
`src/propevolve/training.py`.

These bins are diagnostics, not learned A+ ingredients, but their names make
them look like behavioral rules and they can drift between reports.

Required correction: define one diagnostic headroom-strata specification in
JSON, or compute reported quantiles from the evaluated population. Route every
diagnostic through one shared classifier. Do not feed these bins to the policy
or use them as hard entry gates.

### P1: configuration normalization is not a single seam

`load_experiment_config` injects defaults, while `HistoricalCandidateRunner`
and other downstream modules repeat `.get(..., default)` fallbacks. Direct
runner tests also pass hand-built dictionaries that bypass the loader.

Impact: there are multiple effective configuration implementations. A missing
key can behave differently depending on the call path, and tests can pass while
the real JSON path behaves differently.

The active R19 JSON has one confirmed implicit value:
`training.regime_wait_sequence_fraction = 0.0`.

Required correction:

1. Make one loader return a typed, fully materialized effective recipe.
2. Persist that effective recipe in campaign evidence.
3. Require downstream modules to index normalized fields without local
   fallbacks.
4. Keep legacy migration in a versioned compatibility adapter, separate from
   current-schema validation.
5. Make runner tests enter through the same loader or a shared typed recipe
   fixture.

### P1: artifact layout knowledge is scattered

`src/propevolve/training.py`, `src/propevolve/orchestration.py`, and analysis
modules independently name training recovery, retained policy, diagnostics,
probe, coverage, candidate, and manifest artifacts.

These are stable protocol names rather than run-specific settings, so exposing
every filename as a free-form JSON option would make the interface shallower
and less safe. The duplication is still a maintenance problem.

Required correction: introduce one deep `ArtifactLayout` module selected by a
versioned layout value. JSON selects roots, replay inputs, replay outputs, and
layout version; the module owns stable internal filenames. No caller should
reconstruct those paths.

### P1: launch and local-asset setup contain machine/user conventions

`src/propevolve/launchd.py` embeds the launch label prefix and default macOS log
directory. `src/propevolve/assets.py` embeds local symlink destinations and the
local asset-contract filename.

Required correction: accept these through an operator/runtime configuration or
CLI, with portable product defaults. Generated plist content remains an output;
operators must never edit it.

### P1: the replay exporter reconstructs observation layout with `18`

`scripts/export_recovery_success_replay.py` assumes an
`account_and_management_width` of 18 to recover legacy state.

Impact: an observation-layout change can silently move the realized-PnL index
or invalidate an otherwise reusable replay artifact.

Required correction: read the observation contract stored with the checkpoint
or call a shared observation-layout module. Do not infer a serialized channel
by a numeric width literal.

### P2: current-schema defaults are still hidden in Python

Runtime, replay fractions, epsilon schedules, teacher schedules, target-update
mode, diagnostics interval, reward-shaping values, near-blow fraction, and
reasoning model/effort/timeout have Python defaults.

Defaults are acceptable only in a versioned schema adapter. Current campaign
recipes should serialize them explicitly so the checked-in JSON is the full
source of truth. Once normalized, downstream code must not apply another
default.

### P2: the session roll is fixed at 5:00 p.m. Central

`src/propevolve/environment.py` directly implements the current challenge’s
5:00 p.m. Central session boundary. This matches the present product contract
and should not change during R19.

If PropEvolve will support firms with different session accounting, promote
timezone and session-roll time into the challenge recipe. Otherwise retain this
as an explicit versioned challenge invariant and test it as such.

### P2: production-recipe assertions are mixed into unit tests

Teacher-cache tests load checked-in production JSON and assert exact ticker,
batch-size, date, and output-root values. Those are useful repository contract
checks but are not isolated unit tests.

Required correction: keep one marked recipe-contract test per tracked recipe.
Use temporary semantic fixtures for module unit tests so changing a campaign
recipe does not require unrelated test edits.

## Correct hardcoded contracts to retain

The following should not become arbitrary run settings:

- action enum values and legal action-state transitions;
- serialization schema names and schema versions;
- authenticated manifest fields and SHA-256 requirements;
- teacher channel order and checkpoint-declared architecture compatibility;
- observation tensor layout within a versioned observation schema;
- causal next-bar execution, split boundaries, and teacher-free validation;
- exact pass, blow, and timeout outcome meanings.

Teacher context lengths and suffix lookbacks currently embedded in teacher
adapters are acceptable only as compatibility checks for their frozen artifact
schemas. Future teacher schemas should read these values from authenticated
manifests and reject incompatible artifacts rather than treating them as
campaign tuning options.

## Post-R19 implementation order

1. Freeze R19’s effective recipe, checkpoint/replay identities, and evidence.
2. Add a typed effective-config seam and remove downstream fallback defaults.
3. Make entry supervision genuinely recipe-driven without changing the current
   recipe’s outputs.
4. Replace exact legacy recovery literals with relational validation.
5. Centralize headroom diagnostics without feeding bins to the policy.
6. Introduce the versioned artifact-layout module.
7. Derive replay-export observation indices from the serialized observation
   contract.
8. Move launch and local-asset conventions to runtime/operator configuration.
9. Separate production-recipe contract tests from unit tests.

Each correction must be a separate TDD change. Do not combine it with a new ML
loss, replay policy, curriculum, checkpoint, or economic experiment.

## Acceptance criteria

- A new campaign changes only JSON or CLI values—never Python.
- The current recipe reproduces its existing effective values and identities.
- At least one alternate valid entry and recovery recipe passes validation and
  reaches the same runtime seam without source changes.
- Invalid economic relationships fail closed before loading data or MPS.
- Current-schema JSON contains every behavior-affecting value; the effective
  recipe adds no silent training value.
- No Python source contains concrete campaign IDs, run IDs, checkpoint paths,
  replay selections, episode IDs, ticker universes, fees, or point values.
- Diagnostic strata have one declared source and are never hard policy gates.
- Unit tests do not load mutable production recipes unless explicitly marked as
  repository recipe-contract tests.
- V21 and the current balance curriculum remain backward compatible.
- The full unit suite and one config-only launch preflight pass before any new
  training campaign starts.
