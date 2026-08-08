# AlphaEvolve concepts for PropEvolve

## Scope

This note studies Google DeepMind's [AlphaEvolve white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) and identifies concepts that could strengthen PropEvolve's slow, offline model-improvement process. AlphaEvolve evolves executable programs; PropEvolve trains and promotes trading policies. The mapping below is therefore a design inference, not evidence that AlphaEvolve's results transfer to markets or produce profitable trading.

PropEvolve's current boundary remains fixed: a frozen FFM representation supplies causal market context, recurrent distributional Double-DQN is the initial policy family, completed episodes enter authenticated replay, deployed champions are immutable, and only offline challengers that pass temporal and economic validation may replace a champion.

## Directly supported findings from AlphaEvolve

AlphaEvolve has four interacting parts: a prompt sampler, an ensemble of LLMs, a pool of evaluators, and a program database. A controller samples a parent program and inspirations, builds a prompt containing prior programs and feedback, asks an LLM for a code diff, executes and evaluates the resulting child, and returns the evaluated program to the database. The human supplies the problem, initial solution, and evaluation criteria; the system searches for how to improve it (Sections 2 and 2.1, pp. 3–4).

The following details are especially relevant:

- **Executable evaluation is the grounding mechanism.** Candidate programs receive one or more scalar scores from a fixed evaluator. The paper says this execution-and-evaluation step allows the system to reject incorrect LLM suggestions. It also identifies automated evaluation as AlphaEvolve's main applicability constraint (Sections 1, 2.1, and 6, pp. 2–4 and 21).
- **Prior attempts remain useful inputs.** The program database stores candidates together with scores and outputs, then resurfaces selected programs as parents or inspirations. Its selection mechanism balances improvement of strong programs with diversity, drawing on MAP-Elites and island-based population models (Section 2.5, p. 8).
- **Prompts carry evidence, not merely an instruction.** Prompt context may contain several prior solutions, their rendered evaluation results, explicit domain material, and varied prompt formatting. AlphaEvolve can also evolve prompt-level guidance in a separate database (Section 2.2, pp. 5–6).
- **Candidate generation uses a model mixture.** AlphaEvolve combines a faster model for high-throughput proposals with a more capable model for occasional higher-quality changes (Section 2.3, p. 7).
- **Changes can be constrained to declared surfaces.** Users may mark blocks to evolve while leaving the surrounding program skeleton fixed. Candidate modifications are commonly represented as targeted search-and-replace diffs (Sections 2.1 and 2.3, pp. 4–7).
- **Evaluation can be staged.** An evaluation cascade first tests candidates on smaller or easier cases and spends expensive evaluation only on candidates that pass earlier stages. Independent evaluations can be distributed asynchronously to improve overall throughput (Sections 2.4 and 2.6, pp. 7–8).
- **Multiple metrics help maintain useful diversity.** AlphaEvolve can optimize a vector of scores rather than only one scalar. The authors report that multiple objectives can help even when one metric is primary because different definitions of good produce structurally diverse inspirations (Section 2.4, p. 8).
- **Evolution, context, broad mutation scope, meta-prompts, and model capability all mattered in the studied tasks.** The paper's ablations remove these components separately, and the full method outperforms the reported alternatives on its two ablation tasks (Section 4, pp. 17–18). This supports the components within those tasks; it does not establish their effect in financial ML.
- **Correctness checks remain outside the evolving artifact.** In one systems application, every proposed optimization was checked against reference code on randomized inputs, and the final result received further expert confirmation (Section 3.6, p. 17).
- **The improvement loop is slow relative to execution.** The paper discusses feedback loops for improving AlphaEvolve itself on the order of months and suggests distilling gains into future base models (Section 6, p. 21).

## PropEvolve inferences and recommended adaptations

The following are our proposed adaptations. They are not claims made by the AlphaEvolve authors.

### 1. Use an immutable champion-and-challenger archive

Never overwrite a promoted model. Store every champion and credible challenger as a content-addressed bundle containing:

- policy weights and architecture identity;
- parent model identifiers;
- FFM checkpoint and embedding-contract hashes;
- data, split, simulator, cost, reward, and prop-rule identities;
- training recipe, seeds, code commit, and dependency lock;
- temporal evaluation reports and promotion decision;
- the exact reasoning hypothesis and change from its parent.

This is the closest PropEvolve analogue to AlphaEvolve's program database. It satisfies the requirement that any earlier champion can be restored exactly, while also letting later reasoning learn from unsuccessful branches rather than repeating them.

### 2. Keep a diverse archive instead of one global leaderboard

A single score will favor one behavior and can erase useful alternatives. Maintain a small set of behavior niches, for example:

- highest prop-challenge pass probability;
- lowest MLL blow probability;
- best positive expectancy after costs;
- strongest temporal worst-fold result;
- best Long/Short balance;
- best NQ performance with acceptable cross-market transfer;
- useful trade-frequency bands.

These are research archive descriptors, not live trading rules. They should preserve materially different challengers for future inspiration without deploying an ensemble or mixing their actions.

### 3. Make the economic verifier the source of truth

The LLM may propose and explain experiments, but it must not grade profitability. PropEvolve's evaluator should be executable, deterministic where possible, and return a metric vector that includes at least:

- challenge pass, blow, and timeout rates;
- expectancy and net P&L after fees and slippage;
- maximum and time-under drawdown;
- temporal worst-fold and sealed-confirmation performance;
- Long/Short, market, and period breakdowns;
- action validity, turnover, and abstention behavior;
- calibration or value-distribution diagnostics.

Promotion should remain a fail-closed rule over these metrics, not a freely revised weighted average. The outer reasoning agent may diagnose why a candidate failed, but cannot waive a gate.

### 4. Use a trading-specific evaluation cascade

Adapt AlphaEvolve's cascade into successively more expensive gates:

```text
static contract and lineage checks
  -> deterministic environment and golden-trajectory tests
  -> tiny causal learning smoke test
  -> matched inner temporal evaluation
  -> multi-seed chronological walk-forward
  -> prop challenge Monte Carlo gauntlet
  -> untouched sealed confirmation
  -> shadow deployment
```

The sealed confirmation set must be used only after the recipe is frozen. Unlike an ordinary AlphaEvolve score, every period used to choose a parent, mutation, threshold, or prompt becomes development data and cannot remain a final holdout.

### 5. Give reasoning rich, authenticated experiment context

For each new challenger, provide the reasoning model with a bounded research packet:

- frozen objective and constraints;
- current champion contract and evidence;
- selected diverse prior challengers;
- exact config diffs and evaluator results;
- failure taxonomy and per-slice weaknesses;
- compute budget and allowed revision surface.

This translates AlphaEvolve's context-rich prompt sampling to model research. The system should retrieve relevant attempts by lineage, hypothesis, architecture family, and failure signature rather than dumping the entire history into every prompt.

### 6. Evolve declared experiment surfaces, not arbitrary repository code

AlphaEvolve can evolve entire files, but PropEvolve should initially expose only typed, validated experiment fields such as:

- feed-forward, GRU, or small causal-transformer policy family;
- sequence length, replay curriculum, and sampling balance;
- distributional support and network capacity;
- optimizer and exploration schedule;
- temporary teacher objectives and distillation weights;
- frozen reward variants within the declared prop-economic objective.

Data lineage, temporal splits, sealed holdout, execution timing, costs, prop rules, deployment allowlist, artifact validation, and promotion gates must stay outside the mutation surface. Each accepted proposal should compile to a reviewable JSON diff. Architecture or code mutations can be considered later only inside isolated branches with tests and the same evaluator contract.

### 7. Separate cheap proposal generation from expensive reasoning

A practical model mixture could use a lower-cost reasoning model for routine evidence summarization and syntactically valid config proposals, escalating to a frontier model for novel architecture diagnoses, repeated failures, or genuine blockers. This is an outer-loop compute optimization; it does not imply multiple live trading policies.

### 8. Optimize campaign throughput, but serialize scarce accelerators

Independent CPU validation, report generation, and reasoning requests may run asynchronously. Competing MPS training jobs should remain serialized on the current machine. The goal is higher experiment throughput without corrupting resource availability or creating multiple uncontrolled runs.

### 9. Distill useful guidance, then discard temporary teachers

AlphaEvolve suggests distilling evolution-enhanced capability back into later models. For PropEvolve, a bounded analogue is to use causal out-of-fold Expansion, Pivot, or Trend outputs as temporary training teachers, then evaluate whether the final challenger retains economic lift using only frozen FFM embeddings plus account and execution state. Teacher heads must not become undeclared live dependencies.

## What should not transfer directly

- **Do not let the live agent rewrite its own weights, policy code, reward, or risk rules.** AlphaEvolve searches offline against an evaluator; intraday self-mutation would make the deployed policy unauthenticated and irreproducible.
- **Do not treat backtest maximization as proof.** Financial data are non-stationary, dependent, and vulnerable to repeated-selection overfit. A candidate can exploit simulator or historical-period artifacts while scoring well.
- **Do not reuse the sealed holdout throughout evolution.** Any evaluation result returned to the database or reasoning prompt is selection feedback. That data is no longer sealed.
- **Do not allow LLM judgment to replace economic outcomes.** LLM feedback may assess clarity, novelty, or likely implementation faults, but cannot establish profitability, causality, fill realism, or prop-rule safety.
- **Do not evolve the evaluator and candidate together.** A challenger must not weaken fees, slippage, MLL handling, temporal splits, or pass criteria to improve its own score.
- **Do not deploy a population merely because population search helped AlphaEvolve.** PropEvolve should promote one frozen champion. Archive diversity belongs to offline research unless a separately validated ensemble earns promotion.
- **Do not assume broader code mutation is automatically better.** AlphaEvolve's full-file ablation concerns two mathematical tasks. In PropEvolve, unconstrained code changes enlarge the leakage and evaluator-gaming surface.
- **Do not infer that AlphaEvolve demonstrates profitable trading, robust RL, or safe continual learning.** The paper evaluates algorithm and systems problems with machine-gradeable evaluators, not markets.

## Smallest useful incorporation plan

1. Finish the first standalone PropEvolve training and evaluation path before automating revision.
2. Add immutable, content-addressed model bundles and an append-only lineage index; never replace or delete a champion during promotion.
3. Define the fixed multi-metric economic evaluator and staged evaluation cascade.
4. Record the initial recurrent distributional Double-DQN as the baseline member of the archive.
5. Connect ML Training Loop as an offline proposal-and-evaluation controller with a small allowlisted JSON revision surface.
6. Retain a bounded set of diverse challengers and feed only authenticated evidence into subsequent reasoning.
7. Promote only a challenger that passes matched temporal, prop-economic, and sealed-confirmation gates; otherwise retain the champion and preserve the failed result.
8. Add temporary Expansion guidance only as a matched later experiment, then test teacher-free operation.

## Concrete seams for a later implementation

These interfaces are intentionally smaller and safer than a general program-evolution engine:

```text
CandidateBundle
  identity, parent identities, weights, recipe diff, code/data/checkpoint hashes,
  metric vector, artifacts, terminal decision

CandidateArchive
  add_immutable(bundle)
  get(identity)
  select_inspirations(frozen_research_view, behavior_niches, limit)
  resolve_champion(deployment_market)

ExperimentProposer
  propose(frozen_contract, authenticated_evidence, allowed_revision_schema)
    -> ExperimentRevision

ExperimentRevision
  hypothesis, parent identity, typed JSON diff, expected falsifier,
  compute budget, affected stage

EvaluationCascade
  evaluate(candidate, frozen_contract)
    -> ordered stage receipts + metric vector

PromotionPolicy
  decide(champion, challenger, frozen_gates, sealed_receipt)
    -> PROMOTE | RETAIN | BLOCK

ModelRegistry
  publish_immutable(candidate)
  point_deployment_alias(candidate_identity)
  rollback_to(previous_champion_identity)
```

`CandidateArchive` is evidence storage and research retrieval. `ModelRegistry` is the much smaller deployment authority. Separating them prevents an interesting research candidate from silently becoming tradable.

The frozen research contract should be passed by value or content hash to every seam. No proposal may alter the dataset identity, causal observation rules, split or sealed-holdout definition, simulator and execution assumptions, fees, prop constraints, deployment allowlist, or promotion gates.

## Minimum tests before enabling the loop

- **Immutable artifact test:** publishing a candidate under an existing identity with different bytes must fail.
- **Exact rollback test:** switching the deployment alias back to a prior champion must restore the same weights, manifest, environment contract, and action behavior on golden trajectories.
- **Lineage closure test:** every parent, FFM checkpoint, data receipt, config, code commit, and evaluation artifact referenced by a challenger must resolve and match its recorded hash.
- **Allowed-revision test:** changes outside the typed recipe schema—including risk rules, costs, splits, labels, holdout, and evaluator code—must be rejected before compute.
- **Evaluator determinism test:** identical candidate, data, environment, and seed inputs must yield identical receipts and metrics within declared numerical tolerance.
- **Cascade fail-fast test:** a failed integrity or smoke stage must prevent costly training, gauntlet, and confirmation stages from running.
- **Temporal isolation test:** sealed-confirmation metrics must not appear in proposal prompts, archive sampling features, parent selection, or recipe revisions.
- **Promotion monotonicity test:** no candidate can be deployed without all required receipts and a passing frozen promotion decision.
- **Promotion idempotence test:** replaying the same authenticated promotion event must not create a new model or corrupt the deployment alias.
- **Archive-diversity test:** niche selection should preserve materially different valid candidates and must not reduce to repeatedly selecting the same global winner.
- **Risk-boundary test:** proposed policies cannot bypass the external action mask or deterministic prop-risk enforcement.
- **Crash-recovery test:** interruption at any stage must resume from authenticated completed receipts, without rerunning or overwriting the champion.

## Recommended first increment

Implement only the immutable `CandidateBundle`, append-only `CandidateArchive`, reversible `ModelRegistry`, and the existing evaluator results as authenticated receipts. Do not add autonomous mutations yet. Once the first PropEvolve baseline has completed end to end, add a single bounded proposer that may change one declared training-recipe field and must pass the cascade against the unchanged champion. This proves rollback, evidence flow, and promotion safety before introducing archive sampling, model mixtures, or multiple simultaneous candidate families.

## Bottom line

The strongest AlphaEvolve lesson for PropEvolve is not unrestricted self-modification. It is an evidence-grounded outer loop: retain a diverse history of immutable candidates, propose bounded changes using rich feedback, evaluate them with executable multi-stage tests, and let only verified improvements become future parents. PropEvolve can adopt that architecture while keeping the live trading champion frozen and deterministic risk constraints outside the learned policy. The trading-specific addition is essential: because historical markets are an imperfect evaluator, temporal separation, costs, worst-period evidence, sealed confirmation, and reversible promotion must be stricter than the mechanism described for machine-gradeable algorithmic tasks.
