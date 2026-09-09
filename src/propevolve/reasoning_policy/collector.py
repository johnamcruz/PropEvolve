"""Bounded collection through existing environment and specialist interfaces."""

import numpy as np

from ..decision import Action
from .context import RollingContext
from .dataset import supervised_record, market_supervised_record
from .inputs import specialist_account_fields, observe_context
from .labels import (
    label_actions,
    label_entry_opportunity,
    label_future_excursions,
    label_market_actions,
    label_position_actions,
)


def collect_examples(
    environment, *, reset_options, context_config, sources,
    behavior_factory, continuation_factory, source_id, continuation_id,
    maximum_examples, sample_stride, rollout_max_steps, target_temperature,
    opportunity_contract, collect_action_targets=True, collect_market_targets=True,
    action_label_mode="continuation", collection_warmup_steps=0,
    augment_action_targets=False, initial_entry_action=None,
):
    """Yield market/action records from one declared chronological episode.

Not a full-dataset job: bounded example count avoids accidentally branching
millions of states. The caller owns separate chronological environment slices
and fold-safe specialist source receipts. No caches are rebuilt here.
    """
    if any(type(value) is not int or value < 1 for value in (
        maximum_examples, sample_stride, rollout_max_steps,
    )):
        raise ValueError("collection budgets must be positive integers")
    if type(collection_warmup_steps) is not int or collection_warmup_steps < 0:
        raise ValueError("collection warmup must be a nonnegative integer")
    if not source_id or not continuation_id:
        raise ValueError("source and continuation identities required")
    if not collect_action_targets and not collect_market_targets:
        raise ValueError("collection must request at least one target family")
    if type(augment_action_targets) is not bool:
        raise ValueError("action target augmentation must be boolean")
    if augment_action_targets and not collect_action_targets:
        raise ValueError("action target augmentation requires action records")
    if action_label_mode not in {
            "continuation", "market_barrier_grid", "trade_mastery_grid"}:
        raise ValueError("unknown action label mode")
    if action_label_mode == "trade_mastery_grid":
        try:
            initial_entry_action = Action(initial_entry_action)
        except (TypeError, ValueError) as error:
            raise ValueError("trade mastery requires one Long or Short entry") from error
        if initial_entry_action not in {Action.ENTER_LONG_1, Action.ENTER_SHORT_1}:
            raise ValueError("trade mastery requires one Long or Short entry")
        position_improvement = opportunity_contract.get("position_minimum_improvement_r")
        if (isinstance(position_improvement, bool)
                or not isinstance(position_improvement, (int, float))
                or not np.isfinite(position_improvement) or position_improvement < 0):
            raise ValueError("trade mastery requires a position improvement margin")
    if "ticker" not in reset_options or "start" not in reset_options:
        raise ValueError("collection requires explicit episode identity")
    ticker = reset_options["ticker"]
    market = environment.markets[ticker]
    if environment.spec.per_trade_risk_dollars is None:
        raise ValueError("economic supervision requires the existing declared trade risk")
    observation, info = environment.reset(options=reset_options)
    behavior = behavior_factory()
    history = RollingContext(context_config)
    prefix = []
    row = int(reset_options["start"])
    emitted = 0
    emitted_trade_targets = set()
    entry_action = None
    entry_index = None
    for step in range(rollout_max_steps):
        observe_context(history, environment, observation, ticker=ticker, row=row, sources=sources)
        legal = {Action(value) for value in info["valid_actions"]}
        if (action_label_mode == "trade_mastery_grid" and entry_action is not None
                and legal == {Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1}):
            return
        # Match inference, including initially partial context with its mask.
        sample_due = (step >= collection_warmup_steps
                      and (step - collection_warmup_steps) % sample_stride == 0)
        needs_specialist_targets = bool(
            sources and (collect_market_targets or augment_action_targets))
        specialist_targets_available = (not needs_specialist_targets or all(
            source.targets.target(ticker, row) is not None for source in sources))
        if sample_due and specialist_targets_available:
            window = history.snapshot()
            action_record = None
            if collect_action_targets:
                if (action_label_mode in {"market_barrier_grid", "trade_mastery_grid"}
                        and entry_action is None):
                    expected = {Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1}
                    if legal != expected:
                        raise ValueError("market barrier labels require one flat decision state")
                    labels = label_market_actions(
                        market, decision=row, role_end=len(market.close),
                        observation=observation,
                        risk_dollars=environment.spec.per_trade_risk_dollars,
                        point_value=environment.tick_values[ticker],
                        round_trip_fee=environment.round_trip_fees[ticker],
                        minimum_mll_headroom=info["minimum_mll_headroom"],
                        horizon=opportunity_contract["horizon"],
                        target_rs=opportunity_contract["target_rs"],
                        stop_r=opportunity_contract["stop_r"],
                        utilities=opportunity_contract["utilities"],
                    )
                elif action_label_mode == "trade_mastery_grid":
                    if (legal != {Action.HOLD, Action.CLOSE} or entry_index is None
                            or entry_action is None):
                        raise ValueError("trade mastery requires one positioned decision state")
                    labels = label_position_actions(
                        market, decision=row, role_end=len(market.close),
                        entry_index=entry_index, side=entry_action,
                        observation=observation,
                        risk_dollars=environment.spec.per_trade_risk_dollars,
                        point_value=environment.tick_values[ticker],
                        round_trip_fee=environment.round_trip_fees[ticker],
                        minimum_mll_headroom=info["minimum_mll_headroom"],
                        horizon=opportunity_contract["horizon"],
                        stop_r=opportunity_contract["stop_r"],
                        minimum_improvement_r=position_improvement,
                    )
                else:
                    labels = label_actions(
                        environment, reset_options=reset_options, prefix=tuple(prefix),
                        continuation_factory=continuation_factory, max_steps=rollout_max_steps,
                    )
                action_record = supervised_record(
                    window, labels, source_id=source_id, continuation_id=continuation_id,
                    target_temperature=target_temperature,
                )
                action_record["ticker"] = ticker
            target_grid = None
            excursions = None
            if collect_market_targets or augment_action_targets:
                target_rs = opportunity_contract.get("target_rs")
                if target_rs is None:
                    target_rs = (opportunity_contract["target_r"],)
                target_grid = {f"{float(target):g}": label_entry_opportunity(
                    market, decision=row, role_end=len(market.close),
                    risk_dollars=environment.spec.per_trade_risk_dollars,
                    point_value=environment.tick_values[ticker],
                    round_trip_fee=environment.round_trip_fees[ticker],
                    horizon=opportunity_contract["horizon"], target_r=float(target),
                    stop_r=opportunity_contract["stop_r"],
                ) for target in target_rs}
                opportunity = next(iter(target_grid.values()))
                excursions = label_future_excursions(
                    market, decision=row, role_end=len(market.close),
                    horizon=opportunity_contract["horizon"],
                    risk_dollars=environment.spec.per_trade_risk_dollars,
                    point_value=environment.tick_values[ticker],
                    round_trip_fee=environment.round_trip_fees[ticker],
                )
            else:
                opportunity = None
            specialist_targets = None
            if sources and (collect_market_targets or augment_action_targets):
                teachers = specialist_account_fields(
                    observation, embedding_dim=market.embeddings.shape[1],
                    ticker=ticker, row=row, sources=sources,
                )
                specialist_targets = {
                    key: value for key, value in teachers.items()
                    if not key.startswith("account.")
                }
            if augment_action_targets:
                action_record["targets"].update(
                    target_before_stop_by_r={
                        key: {"long": value[0], "short": value[1]}
                        for key, value in target_grid.items()
                    },
                    future_excursions=excursions,
                )
                if specialist_targets is not None:
                    action_record["targets"]["specialist_targets"] = specialist_targets
            market_record = None
            if collect_market_targets and opportunity is not None:
                market_record = market_supervised_record(
                    window, opportunity=opportunity, source_id=source_id,
                    label_end_ns=int(market.timestamps[row + opportunity_contract["horizon"]].astype("datetime64[ns]").astype(np.int64)),
                    excursions=excursions,
                    economic_contract={
                        **opportunity_contract,
                        "risk_dollars": environment.spec.per_trade_risk_dollars,
                        "point_value": environment.tick_values[ticker],
                        "round_trip_fee": environment.round_trip_fees[ticker],
                        "execution": "next_bar_open", "same_bar_collision": "adverse_first",
                    },
                    target_grid=target_grid,
                )
                market_record["ticker"] = ticker
                if context_config.input_mode == "embeddings":
                    import json
                    market_record["targets"]["specialist_targets"] = specialist_targets
                    market_record["messages"][-1]["content"] = json.dumps(market_record["targets"], allow_nan=False)
                    market_record["messages"][0]["content"] += " Also estimate the labeled specialist market state."
            target_name = (None if action_record is None else
                           action_record["messages"][-1]["content"])
            emit_record = (action_label_mode != "trade_mastery_grid"
                           or target_name not in emitted_trade_targets)
            if emit_record:
                yield {"action": action_record, "market": market_record}
                emitted += 1
                if target_name is not None:
                    emitted_trade_targets.add(target_name)
                if emitted >= maximum_examples:
                    return
        action = (initial_entry_action
                  if (action_label_mode == "trade_mastery_grid"
                      and entry_action is None and sample_due)
                  else behavior(observation, info))
        observation, _, terminated, truncated, info = environment.step(action)
        prefix.append(action)
        row = int(info["fill_index"])
        if Action(action) in {Action.ENTER_LONG_1, Action.ENTER_SHORT_1}:
            entry_action = Action(action)
            entry_index = row
        if terminated or truncated:
            return
    raise ValueError("collection episode incomplete within budget")
