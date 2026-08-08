# Stanford CS329A concepts for a self-improving trading RL agent

## Scope

This note studies Stanford CS329A's self-improving-agent curriculum and maps its most relevant ideas to PropEvolve. The factual findings below come from the official lectures, course syllabus, and primary papers. The proposed trading design is an inference: the cited work does not demonstrate profitable trading or direct transfer to financial markets.

## Directly supported findings

The first linked video is Stanford CS329A's [Part 1: Course Overview](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=116s). The [official course syllabus](https://cs329a.stanford.edu/) defines self-improvement as improvement through interaction with the system itself and its environment, then organizes the subject around test-time search, verification, environmental feedback, multi-step planning, RL, memory, open-ended evolution, and long-horizon evaluation. The [official playlist](https://www.youtube.com/playlist?list=PLangBM27OtEA) contains the lecture series.

Together, the supplied recordings contribute five additional views of the loop:

- [Part 3: Robust Verification](https://www.youtube.com/watch?v=p7TdPUcPoik): generating alternatives is useful only when the system can reliably distinguish good outputs from bad ones. [Large Language Monkeys](https://arxiv.org/abs/2407.21787) finds that repeated sampling increases the probability that at least one candidate is correct, but selection with imperfect reward models or voting eventually plateaus. [Weaver](https://arxiv.org/abs/2506.18203) improves candidate selection by normalizing and combining multiple imperfect verifiers.
- [Part 4: Learning from Feedback with Tools/Code](https://www.youtube.com/watch?v=Lxh9RF5S-K0): [ReAct](https://arxiv.org/abs/2210.03629) interleaves reasoning, action, and new observations so plans can be updated rather than executed blindly. [RLEF](https://arxiv.org/abs/2410.02089) uses public and private execution-test feedback in a PPO loop and teaches models to improve across attempts instead of resampling independently. The lecture contrasts environmental feedback, executable feedback, and model-generated critique; those feedback sources are not equally trustworthy.
- [Part 5: Planning and Multi-Step Reasoning](https://www.youtube.com/watch?v=Ml_fp9XkB8Y&list=PLangBM27OtEA&index=5): the lecture covers planning over multiple actions, environmental feedback, and step-level learning. [LATS](https://arxiv.org/abs/2310.04406) combines tree search, value estimates, self-reflection, and external feedback. [SWiRL](https://arxiv.org/abs/2504.04736) decomposes multi-step trajectories into action-level sub-trajectories and applies filtering plus step-wise RL.
- [Part 7: Self-Improvement and Deep Research Agents](https://www.youtube.com/watch?v=Uni9dqyuuDM): [AlphaCode](https://arxiv.org/abs/2203.07814) samples many programs, execution-filters them, clusters candidates by behavior, and submits a small diverse set. It temporally separates training from evaluation and reserves hidden tests for final evaluation. Its paper also reports that validation loss is a poor proxy for solve rate. [AlphaCode 2](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf) adds diverse policy models and a learned scoring model for reranking candidates after executable filtering. [Search-o1](https://arxiv.org/abs/2501.05366) triggers external retrieval when the reasoning process encounters uncertainty and refines retrieved material before using it, rather than injecting all available memory continuously.
- [Part 8: Agentic Evaluations and Long Horizon Tasks](https://www.youtube.com/watch?v=8JAqLnTaZu4): [METR's time-horizon work](https://arxiv.org/abs/2503.14499) evaluates success at declared reliability levels as task duration grows and attributes stronger long-horizon results partly to reliability and recovery from mistakes. [GDPval](https://arxiv.org/abs/2510.04374) tests real economically valuable work and reports gains from reasoning effort, context, and scaffolding. [DeepScholar-Bench](https://arxiv.org/abs/2508.20033) separates synthesis, retrieval quality, and verifiability. The lecture also identifies recurring failure modes such as poor planning, wrong tool/action selection, premature abandonment, and repetitive loops.

Other course readings supply complementary pieces:

- [STaR](https://arxiv.org/abs/2203.14465) repeats a generate, verify, retain-successful-traces, and retrain cycle. Its evidence concerns reasoning tasks, not markets.
- The [Darwin Godel Machine](https://arxiv.org/abs/2505.22954) preserves an archive of diverse agents, proposes modifications, and empirically validates each change in a sandbox with human oversight. [AlphaEvolve](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) similarly couples candidate generation with automated evaluation and evolution.
- [MemGPT](https://arxiv.org/abs/2310.08560) separates fast and slow memory tiers to preserve useful context beyond a model's immediate context window.

Three kinds of "memory" must not be conflated:

| Mechanism | What it is | Lifetime | Role in learning or inference |
|---|---|---|---|
| Recurrent state | A learned hidden state, such as an LSTM/GRU state, summarizing the causally observed sequence | Usually one episode; reset at a declared boundary | Gives the policy temporal context at the next decision |
| Experience replay | Stored transitions or sequences sampled again for gradient updates | Across training updates | Improves sample use for compatible off-policy methods; recurrent replay must preserve sequences and reconstruct hidden state |
| External episodic memory | An explicit store of prior episodes retrieved by similarity or keys | Across episodes and potentially deployments | Supplies particular past cases directly at decision or analysis time |

[R2D2](https://openreview.net/pdf?id=r1lyTjAqYX) illustrates the interaction between recurrent state and replay: it stores fixed-length sequences, never crosses episode boundaries, and unrolls recurrent networks over the replayed sequence. [Neural Episodic Control](https://arxiv.org/abs/1703.01988) is a distinct external-memory design that stores state representations and rapidly updated value estimates for context-based lookup. [PPO](https://arxiv.org/abs/1707.06347), which algoTraderAI currently uses, is principally an on-policy method that alternates environment sampling with several minibatch optimization epochs; an unrestricted historical replay buffer cannot simply be added without changing the learning assumptions.

## Existing seams worth reusing

algoTraderAI already has much of the grounded environment needed for a first PropEvolve experiment:

- [`PropFirmEnv`](https://github.com/johnamcruz/algoTraderAI/blob/main/rl/environments/prop_firm.py) advances to actual decision points and exposes account, position, signal, regime, and combine-progress observations.
- Its account observations include balance, equity, drawdown, distance to the maximum-loss-limit floor, and remaining episode time. [`compute_dmll`](https://github.com/johnamcruz/algoTraderAI/blob/main/utils/account_mll.py) defines the live/simulated risk-budget calculation.
- The environment's verified feedback includes realized reward, drawdown pressure, pass, timeout, and MLL blow outcomes.
- [`pipelines/ffm/backbone.py`](https://github.com/johnamcruz/algoTraderAI/blob/main/pipelines/ffm/backbone.py) already treats the FFM representation as a frozen embedding source.

The present environment mostly exposes downstream signal scores rather than the full frozen FFM embedding. PropEvolve's proposed observation should therefore be explicit:

`state_t = [frozen FFM market embedding_t, account/MLL state_t, position state_t, action/fill history_t]`

This preserves the foundation model as a frozen market observer while allowing the RL policy to learn account-aware action selection. It also keeps representation learning separate from the trading objective.

## Recommended PropEvolve interpretation

The user's loop is sound if split into two timescales:

```text
Fast execution loop (weights frozen)
observe causal state
  -> score every valid action
  -> apply deterministic prop-risk constraints
  -> execute one action or abstain
  -> record fills, costs, reward, and next state

Slow improvement loop (offline only)
authenticate completed trajectories
  -> diagnose errors and successes
  -> train one candidate policy
  -> temporal walk-forward + costs + MLL gauntlet
  -> promote only a verified challenger
  -> otherwise retain the champion
```

The lectures imply a verifier hierarchy for PropEvolve:

1. Broker or causal simulator truth—fills, costs, P&L, drawdown, MLL blow, pass, and timeout—is the outcome verifier.
2. Deterministic prop-risk rules are pre-execution validity checks and cannot be learned away.
3. A learned critic or scoring model is a candidate ranker. It must be calibrated against later economic outcomes and is not ground truth.
4. LLM-generated critique can diagnose experiments and propose revisions, but it must never become the trading reward or override the first two layers.

For the initial action space, exact enumeration remains better than AlphaCode-style massive sampling. At a decision point, the valid set is small: abstain, long/short at allowed sizes while flat, or hold/close/manage while in a position. The actor can propose probabilities and the critic can estimate each action's account-conditioned value. A deterministic risk layer must reject actions that violate prop constraints. Diversity and clustering become useful in the slower outer loop, where several materially different policy candidates may otherwise duplicate the same behavior. Tree search becomes justified only if we later have a validated short-horizon market/fill model; otherwise imagined future prices are an unreliable verifier.

"See result" should mean simulator or broker truth after commissions, slippage, fills, realized and unrealized P&L, drawdown, and MLL effects. An LLM critique is useful for research diagnosis, but it is not a profitability verifier and must not override economic evidence.

"Analyze and remember" should initially produce an authenticated structured episode record containing:

- all information available at the decision time, including the frozen FFM embedding hash and account state;
- candidate actions, chosen action, action probability, and predicted value;
- fill price, costs, MFE/MAE, realized return, drawdown, MLL impact, and terminal cause;
- policy, environment, data, and reward-contract identities.

For the first POC, store these records but do not retrieve them into live inference. Train a recurrent PPO policy on fresh causally sampled rollouts and test whether recurrent state adds value over a feed-forward PPO baseline. RLEF supports the narrower claim that execution feedback can train a PPO agent over multiple attempts when the execution verifier is reliable. It does not make old arbitrary trajectories on-policy. Historical episodes can drive diagnostics and episode sampling, but should not be treated as an unrestricted replay buffer. If direct replay becomes a requirement, test a deliberately off-policy recurrent learner as a separate matched experiment.

External episodic retrieval is a later ablation. Search-o1 suggests a better form than always-on retrieval: query memory only when policy uncertainty or critic disagreement crosses a frozen threshold, refine the retrieved episodes to a compact structured summary, and allow abstention if uncertainty remains high. A query could retrieve prior states similar in frozen FFM embedding and account/MLL context, but historical evaluation must expose only episodes completed before the current decision timestamp. It must beat the no-retrieval policy under chronological walk-forward; otherwise it adds complexity and a leakage surface without evidence.

Part 8 changes evaluation more than architecture. PropEvolve should measure pass and survival reliability over progressively longer account windows, not just mean episode reward. It should also emit a failure taxonomy for wrong-side entry, bad size, premature close, excessive abstention, repeated action loops, and risk-rule rejection. These diagnostics can guide the offline improvement loop without becoming handcrafted live-policy rules.

### Does the added material change the initial architecture?

No. It strengthens four boundaries rather than adding a new live component:

- reward remains verified account economics, not model self-critique;
- candidate selection remains exact action enumeration plus a learned critic because the action set is small;
- recurrent PPO remains the first temporal-memory test, while replay and external episodic retrieval stay separate later ablations;
- planning remains one-step action selection until a causal market/fill world model earns the right to support tree search.

The only addition to the initial POC is evaluator-side: record verifier/ranker disagreement, action-loop failures, and reliability by account-window length. AlphaCode's hidden-test design reinforces keeping sealed temporal data entirely outside candidate filtering and policy revision.

## Smallest credible experiment

1. Freeze the FFM checkpoint, embedding contract, market data, prop rules, costs, temporal splits, and evaluation gates.
2. Reuse the algoTraderAI environment and account/MLL state, adding the frozen FFM embedding as market state.
3. Compare feed-forward PPO with recurrent PPO using the same actions, reward, rollouts, seeds, and compute.
4. At each state, evaluate all valid actions; keep hard risk constraints outside the learned policy.
5. Measure pass probability, MLL blow rate, expectancy after costs, drawdown, turnover, calibration, and per-period/per-market robustness. Report success at multiple account-window lengths and reliability levels, plus the declared failure taxonomy.
6. Keep a sealed final period. Do not let episode reflection, memory retrieval, or the outer improvement loop revise against that period.
7. Only after recurrent PPO shows matched temporal lift, test either sequence replay with an off-policy learner or external episodic retrieval—one mechanism at a time.

## Bottom line

The most useful idea from CS329A is not to let a live trading agent rewrite itself after each trade. It is to combine candidate generation with a trustworthy verifier and a slower evidence-gated improvement loop. For PropEvolve, market/account observations and the policy's recurrent state belong in the fast loop; authenticated trajectories, replay, diagnosis, retraining, and promotion belong in the slow loop. That design can learn from every trade while keeping live behavior frozen, causal, reproducible, and bounded by the prop firm's loss constraints.
