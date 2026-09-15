# Domain context

- **Entry decision**: ENTER means taking a trade; WAIT means sitting out.
  The configured reasoning model assesses entry separately from direction using its predicted market
  interpretation. ENTER is not an executable sixth action.
- **Market interpretation**: the reasoning model's learned description of causal market
  conditions from Chronos embedding history, supervised by authenticated
  Expansion, Trend, Regime and Volume targets during training only.
- **Trade assessment**: the reasoning model's learned entry, direction and management
  judgments based on its own market interpretation and causal trade state.
  Teacher answers and future trade outcomes are not assessment inputs.
- **Direction decision**: when the economic label calls for ENTER, evaluate
  LONG versus SHORT separately. Entering on the wrong side fails direction and
  the complete trade action, but does not also fail the ENTER decision.
- **Management decision**: an open position learns HOLD versus CLOSE only.
- **Executable management labels**: JSON opportunity contract
  `management_label_mode: simulator_continuation` compares next-open CLOSE with
  an injected causal continuation using the unchanged simulator. Stops and
  trailing ratchets retain the environment's execution order. The configured
  horizon is a finite labeling budget with a final CLOSE, not a 2R/3R/4R profit
  ceiling. Existing recipes without this field retain their old oracle-potential
  semantics and must not be presented as executable continuation evidence.
  `targets.management_evidence` records branch MFE/MAE, realized net R and exit
  reasons for audit only; those future receipts never enter the policy context.
  Running MFE/MAE use gross initial-stop-distance R; realized net R uses dollar
  risk after fees. The record explicitly names both units. Challenge-forced
  exits are censored rather than substituted for trade-mastery supervision.
- **Partial mastery**: retain each correct applicable boundary and correct each
  incorrect one independently. WAIT targets have no direction supervision.
- **Trading evidence**: future target-before-stop and excursion outcomes are
  labels, never policy inputs. Teacher-free policy evaluation must use the same
  decision definitions as training, retention, and checkpoint selection.
- **Management collection coverage**: job JSON `management_sampling: all_states`
  preserves repeated HOLD/CLOSE labels at the configured `sample_stride`, up to
  `maximum_examples_per_episode`. `first_per_action` retains legacy collection
  behavior. Both explicit and economic-sampling episode paths use the same
  trade-management plan. Flat WAIT remains a single entry decision. Rebuilding
  a corpus requires a new output and coverage audit; existing assessments are
  not evidence for newly collected rows. Repeated neighboring rows are not
  independent evidence of generalization.
- **Prospective trade R context**: `trade.volatility_r` is the arithmetic mean
  true range over the JSON-configured number of completed bars, multiplied by
  contract point value and divided by configured dollar risk.
  `trade.cost_r` is round-trip fees divided by that same risk. A full lookback
  plus preceding close is required; `trade.volatility_available` distinguishes
  missing history from zero volatility. These inputs are available while flat
  and are trade economics, not challenge balance or MLL. Existing open-trade
  excursion fields retain their original initial-stop-distance denominator.
  A context JSON may declare `text_fields` to keep these appended R inputs
  continuous-only. Dataset `causal_state_fields` names the numeric values;
  visible prompt values must match exactly. This preserves the original text
  while supplying the same numeric state in prepared training and live policy
  inference. New projector columns initialize at zero when explicitly extending
  a parent; matched diagnostics verify initial action-score parity before updates.

- **Challenge P&L**: net realized or marked-to-market profit relative to the
  challenge starting balance; the model does not depend on the broker's
  absolute balance coordinate.
- **MLL floor**: the challenge P&L level at or below which the account blows.
- **EOD trail**: at the 5:00 p.m. America/Chicago session boundary, the MLL
  floor ratchets to at most one maximum-loss allowance below realized P&L and
  never moves backward.
- **Passmark lock**: once realized P&L reaches one maximum-loss allowance, the
  MLL floor locks permanently at zero challenge P&L.
- **Pass**: net challenge P&L reaches the configured profit target.
- **Blow**: intrabar net equity touches or crosses the effective MLL floor;
  blow has priority over a recovering bar close.
- **Timeout**: the episode window ends without a pass or blow; any open
  position is liquidated with costs.
- **Golden trajectory**: an immutable input/action sequence and expected
  economic receipt captured from the trusted simulator contract. PropEvolve
  must reproduce it through its public `reset`/`step` interface.
### Management-only diagnostic lineage

Trade-mastery management rows retain actual entry side and execution-bar timestamp
in `targets.position_entry`, including when `management_only` excludes the entry
label. This is audit metadata, not a policy prompt input. The independent OHLC
audit resolves legacy entry rows through authenticated selection-source manifests
and rejects missing/conflicting entry lineage rather than guessing from WAIT.
The bounded management correction uses JSON-selected rows, frozen-parent inference
first, and the unchanged chronological control. Collection/audit success is not
proof of learning, retention, or economic generalization.
