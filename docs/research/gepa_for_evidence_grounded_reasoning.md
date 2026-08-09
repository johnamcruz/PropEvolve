# GEPA concepts for evidence-grounded ML reasoning

## Scope and verdict

This note studies GEPA from its official paper and repository and asks whether
it can improve the shared ML Training Loop's diagnosis-and-proposal workflow for
PropEvolve. The recommendation is **yes, narrowly**: use GEPA offline to optimize
one or more frozen textual reasoning instructions against a curated set of prior
development-only reasoning checkpoints. Do not place GEPA inside a live campaign,
let it authorize revisions, or let it optimize PropEvolve's policy, reward,
evaluator, temporal roles, or economic gates.

GEPA's reusable contribution is a search method for textual parameters. It turns
execution and evaluator traces into natural-language feedback, asks a reflection
model to propose a targeted text mutation, tests the mutation on the same
minibatch, and retains candidates that improve. A per-example or per-objective
frontier preserves different successful strategies for later mutation. This is a
strong conceptual fit for improving a reasoning prompt that must diagnose a failed
experiment and produce one bounded revision. It is not evidence that repeated
optimization against historical trading runs will improve future economics.

| Adopt | Keep unchanged |
|---|---|
| Authenticated actionable side information | ML-RIGOR temporal and economic gates |
| Offline reflective mutation of versioned reasoning text | One bounded runtime `NEEDS_REASONING` decision |
| Paired candidate evaluation and lineage | Host-owned schema, revision validator, and authorization |
| Search diversity across diagnosis families | Frozen policy, reward, risk rules, data, splits, and sealed period |

For PropEvolve candidates, zero blow remains a hard feasibility condition rather
than a large term in an aggregate GEPA score: first reject any candidate with a
blow, then rank eligible candidates by pass rate and secondary timeout/economic
metrics. GEPA's default summed minibatch acceptance could otherwise trade one
blow for enough improvements elsewhere, and its per-key frontier can preserve a
specialist that is unsafe on another market or period.

## What GEPA actually optimizes

The ICLR 2026 paper defines a compound AI system with prompts, model weights, and
control flow, but GEPA itself evolves only the prompts while keeping the underlying
LLM weights frozen. Each iteration selects an existing candidate, samples a
training minibatch, records the program trajectory and evaluator feedback, updates
one prompt component through LLM reflection, accepts the child only if its
minibatch score improves, and evaluates accepted children on a selection set. The
returned candidate is the one with the best aggregate selection-set score
([paper, Sections 2-3 and Algorithms 1-2](https://arxiv.org/pdf/2507.19457)).

The current repository generalizes the candidate from prompts to any named text
components. Its adapter contract still represents a candidate as
`dict[str, str]`; the host executes it, supplies per-example numeric scores and
optional trajectories, and formats a small reflective dataset for the proposal
model. GEPA treats trajectories and outputs as opaque and uses the host's scores
for acceptance and selection
([adapter contract at inspected commit `8a2bed9`](https://github.com/gepa-ai/gepa/blob/8a2bed96385202f69caaeb5327a843ed2f5ea225/src/gepa/core/adapter.py#L81-L140)).

Therefore GEPA does not, by itself:

- train or select PropEvolve's recurrent distributional Double-DQN;
- determine whether a diagnosis is scientifically correct;
- protect causal or temporal boundaries;
- distinguish a safe timeout policy from an over-conservative policy;
- establish profitability, future challenge pass probability, or blow safety;
- decide which experiment revision is authorized.

All of those properties must come from PropEvolve's authenticated evidence,
fixed validators, temporal evaluation, and the host-owned reasoning policy.

## Reusable ideas

### 1. Actionable side information instead of metric-only feedback

GEPA pairs the scalar result with execution traces and evaluator-produced text,
such as failed rubrics or compiler errors. The paper argues that this lets the
reflection model perform implicit credit assignment and propose targeted changes
rather than learning only that a candidate lost
([paper, Section 3](https://arxiv.org/pdf/2507.19457)).

For the ML Training Loop, the analogous side information should be an
authenticated, bounded diagnostic record rather than raw logs:

- failed stage and first failed boundary;
- exact gate decision, thresholds, and evidence identifiers;
- model and economic deltas against the matched parent;
- per-market, temporal-fold, side, pass/blow/timeout, and calibration slices;
- integrity and parity checks already passed or failed;
- prior hypothesis, exact config diff, and observed falsifier;
- immutable fields and the allowlisted revision surface.

This does not require GEPA at runtime. Improving the shape of this reflective
record is useful independently and is the lowest-risk concept to adopt.

### 2. Optimize components, not one undifferentiated prompt

GEPA can treat a workflow as named textual components and update a selected
subset. The adapter receives `components_to_update`, and its reflective dataset
is organized per component
([adapter contract](https://github.com/gepa-ai/gepa/blob/8a2bed96385202f69caaeb5327a843ed2f5ea225/src/gepa/core/adapter.py#L183-L214)).

The reasoning workflow has natural components that can remain independently
versioned:

1. evidence interpretation and integrity triage;
2. first-failed-boundary diagnosis;
3. one-change experimental proposal and falsifier;
4. structured `REVISE | STOP | BLOCK` output discipline.

The first trial should optimize only the diagnosis-and-proposal instruction.
Evidence serialization, response schema, revision validation, and authorization
must remain fixed host code.

### 3. Paired before/after evaluation

GEPA evaluates the parent and mutation on the same sampled cases and its default
acceptance policy requires a strict increase in the sum of the new scores
([acceptance implementation](https://github.com/gepa-ai/gepa/blob/8a2bed96385202f69caaeb5327a843ed2f5ea225/src/gepa/strategies/acceptance.py#L42-L54)).
That paired design is appropriate for prompt comparison: run the same model,
reasoning effort, evidence packet, tool availability, and seed policy under the
baseline and candidate prompt. It does not remove model sampling noise, so prompt
selection should use repeated fixed seeds where supported and report paired
case-level outcomes rather than only an aggregate mean.

### 4. Preserve several kinds of successful reasoning

GEPA's default frontier is not a conventional continuous Pareto frontier. It
tracks which candidates are best for individual examples or named objectives and
samples candidates in proportion to how many frontier keys they lead. The
official guide explicitly warns that a well-balanced candidate that is never best
on any key may be excluded
([candidate-selection guide](https://gepa-ai.github.io/gepa/guides/candidate-selection/)).

For reasoning-prompt research, useful frontier keys could include failure family
(integrity, non-learning, calibration, economics, or parity), model family, and
hard policy compliance. These are reasoning-evaluation strata, not trading
metrics or deployment niches. The final prompt still needs one frozen aggregate
selection rule and an unseen confirmation set.

### 5. Retain lineage and lessons from failed branches

GEPA retains candidates, parents, scores, and proposal history so later
reflection can build on successful branches rather than repeatedly editing one
global best. ML Training Loop already has the stronger primitive: immutable
receipts, an authenticated trial ledger, prior revisions, and bounded reasoning
checkpoints. A GEPA experiment should reference those records; it should not
create a second source of campaign truth.

## The safest integration seam

The seam is an **offline prompt-development harness around the existing reasoning
adapter**, not a new node in the campaign state machine.

```text
authenticated historical development receipts
  -> fixed reasoning-case builder
  -> baseline or candidate reasoning instruction
  -> existing reasoning model, effort, tools, and JSON schema
  -> deterministic validators plus curated diagnosis labels
  -> per-case score and actionable feedback
  -> GEPA prompt search on development cases only
  -> frozen winner selected on a separate chronological set
  -> shadow audit on later unseen reasoning checkpoints
  -> explicit human promotion of the prompt artifact
```

The ML Training Loop should continue to own the evidence envelope, response
schema, revision budget, revision validator, receipts, and final authorization.
PropEvolve should continue to own candidate bundles, economic evaluation,
temporal roles, simulator identity, and allowed configuration fields. GEPA owns
only the offline search over the text inserted by the host's prompt builder.

Do not initially use the repository's broad `optimize_anything` agent mode for
this dataset. At the inspected commit its evaluation server deliberately exposes
the combined train and validation example pool to the optimizing agent, while
holding only the test set outside the server
([evaluation-server contract](https://github.com/gepa-ai/gepa/blob/8a2bed96385202f69caaeb5327a843ed2f5ea225/src/gepa/oa/eval_server.py#L1-L29),
[visible-pool implementation](https://github.com/gepa-ai/gepa/blob/8a2bed96385202f69caaeb5327a843ed2f5ea225/src/gepa/oa/eval_server.py#L597-L609)).
That is incompatible with using a selection period as if it were still unseen.
A custom adapter around the core optimizer, or an equivalent small internal
harness, must enforce the project's chronological roles.

## Evaluating a reasoning prompt

The evaluator must not ask another LLM for a single holistic quality score and
treat that as truth. Use hard checks first and human- or outcome-grounded labels
where judgment is unavoidable.

### Fail-closed checks

A response scores zero and is ineligible if it:

- is not valid against the existing `REVISE | STOP | BLOCK` schema;
- changes the wrong stage or proposes more than one revision;
- touches data identity, labels, temporal roles, sealed evidence, evaluator,
  simulator, costs, prop rules, promotion gates, or another frozen field;
- references evidence not present in the authenticated packet;
- omits a causal hypothesis, exact config diff, or explicit falsifier;
- claims that training loss, model self-critique, or inspected holdout lift is
  promotion evidence.

### Scored evidence after hard checks

Use a small explicit vector rather than one vague quality number:

- first-failed-boundary diagnosis agrees with a curated adjudication;
- cited evidence identifiers support every material claim;
- proposed diff passes the host revision validator;
- proposal directly tests the diagnosis and changes only one causal factor;
- stop/block decisions agree with exhausted-budget or integrity-block labels;
- concise completeness and token cost;
- where already available, the downstream matched development result supported
  the proposal's declared falsifier.

Downstream result must be used carefully. A good proposal can be falsified, and
a lucky candidate can improve noisy economics for the wrong reason. Score the
scientific quality of the test separately from its outcome. Any market period or
campaign outcome returned to GEPA becomes development data and cannot later be
called sealed confirmation.

## Temporal-trading risks

- **Selection leakage:** GEPA repeatedly selects against its validation scores.
  In trading ML, every exposed period, market, fold, episode, and failure slice
  becomes part of prompt selection. Randomly splitting rows or episodes does not
  restore independence.
- **Evaluator gaming:** a prompt can learn to satisfy schema checks, echo gate
  language, or choose safe `STOP` decisions without improving diagnosis. Hard
  compliance and diagnosis quality must be reported separately.
- **Over-conservatism:** optimizing heavily for zero invalid revisions can reward
  always stopping, analogous to a policy that avoids blowing but never passes.
  The evaluator needs coverage and useful-proposal recall subject to the hard
  safety constraints.
- **Noisy delayed credit:** whether one ML revision helped may require a costly,
  multi-seed temporal study. It is not the dense, deterministic evaluator that
  makes GEPA strong on code or rubric tasks.
- **Repeated-research overfit:** even without sealed data, repeatedly optimizing
  prompts against the same development campaigns can encode period-specific
  folklore. Split at the campaign or chronological study level, not by receipt
  row.
- **Frontier mismatch:** GEPA's best-per-key frontier can favor brittle
  specialists and exclude a consistently good prompt. Treat the frontier as a
  search-diversity mechanism, never as the promotion rule.
- **Prompt/model coupling:** an optimized instruction may depend on the exact
  reasoning model, effort, tools, or output parser. Freeze and record all of
  those identities and revalidate after any model upgrade.
- **Sensitive trace exposure:** reflective records may contain local paths,
  credentials, proprietary strategy details, or sealed metrics. Reuse the
  existing redaction and authenticated-packet boundary before any external model
  call.

## Minimal non-intrusive adoption plan

1. **Make no change to the current PropEvolve campaign.** Preserve its existing
   reasoning adapter, one-revision limit, economic gates, and sealed 2026 period.
2. **Version the current reasoning instruction as the baseline artifact.** Record
   its text hash, model, effort, tools, schema, prompt builder, and revision
   validator identities.
3. **Build a development-only reasoning-case set from completed receipts.** Use
   only periods and campaigns already designated for research. Group cases by
   campaign and time; create chronological reflection, selection, and untouched
   confirmation partitions. Keep all sealed PropEvolve evidence absent.
4. **Adjudicate the smallest representative set.** Label the first failed
   boundary, allowable decision, supporting evidence, and acceptable one-change
   falsifier. Include safety, over-conservative timeout, non-learning,
   calibration, and economics cases.
5. **Run one bounded offline GEPA comparison.** Optimize only the
   diagnosis-and-proposal instruction, use strict improvement, disable merge,
   cap metric calls and reflection cost, and keep the existing schema and
   validator outside the mutation surface.
6. **Select once on the chronological selection partition.** Require zero hard
   violations and matched lift over the baseline across diagnosis families;
   report paired uncertainty and token cost. Do not tune after viewing the
   untouched reasoning confirmation results.
7. **Shadow before promotion.** On later unseen `NEEDS_REASONING` checkpoints,
   run baseline and candidate without allowing the GEPA-derived output to launch
   work. Compare both to the final authorized decision and subsequent evidence.
8. **Promote explicitly or discard.** If the candidate passes the frozen prompt
   gates, publish its immutable text and evaluation receipt and update the host's
   prompt version deliberately. Otherwise retain the current prompt and preserve
   the negative result.

## Recommendation

Adopt GEPA's **feedback representation and offline prompt-search pattern**, not
its optimizer as a new autonomous controller. The likely first gain is better
structured reasoning packets and fewer generic or repeated diagnoses. A GEPA
dependency is optional: the adapter contract and reflective-dataset format can
be prototyped first using the existing ML Training Loop receipts, and the actual
search library should be introduced only if a baseline-versus-GEPA prompt study
has a trustworthy evaluator and enough independent historical reasoning cases.

The decisive gate is not whether GEPA finds a higher development score. It is
whether a frozen GEPA-derived instruction produces more correct first-boundary
diagnoses and more useful, policy-valid one-change experiments on unseen
reasoning checkpoints, without increasing unsafe revisions or consuming sealed
trading evidence.
