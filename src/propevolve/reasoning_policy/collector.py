"""Bounded collection through existing environment and specialist interfaces."""

import numpy as np

from .context import RollingContext
from .dataset import supervised_record, market_supervised_record
from .inputs import specialist_account_fields
from .labels import label_actions, label_entry_opportunity


def collect_examples(
    environment, *, reset_options, context_config, sources,
    behavior_factory, continuation_factory, source_id, continuation_id,
    maximum_examples, sample_stride, rollout_max_steps, target_temperature,
    opportunity_contract,
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
    if not source_id or not continuation_id:
        raise ValueError("source and continuation identities required")
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
    for step in range(rollout_max_steps):
        fields = specialist_account_fields(
            observation, embedding_dim=market.embeddings.shape[1], ticker=ticker,
            row=row, sources=sources,
        )
        fields.update(environment.causal_trade_context())
        # Selecting a declared subset allows optional production management
        # coordinates, without allowing a future label field into the adapter.
        if not set(context_config.fields).issubset(fields):
            raise ValueError("configured input is not available from causal sources")
        history.append(
            int(np.datetime64(info["timestamp"], "ns").astype(np.int64)),
            {key: fields[key] for key in context_config.fields},
        )
        if history.snapshot().available.all() and step % sample_stride == 0:
            window = history.snapshot()
            labels = label_actions(
                environment, reset_options=reset_options, prefix=tuple(prefix),
                continuation_factory=continuation_factory, max_steps=rollout_max_steps,
            )
            action_record = supervised_record(
                window, labels, source_id=source_id, continuation_id=continuation_id,
                target_temperature=target_temperature,
            )
            opportunity = label_entry_opportunity(
                market, decision=row, role_end=len(market.close),
                risk_dollars=environment.spec.per_trade_risk_dollars,
                point_value=environment.tick_values[ticker],
                round_trip_fee=environment.round_trip_fees[ticker],
                **opportunity_contract,
            )
            market_record = None
            if opportunity is not None:
                market_record = market_supervised_record(
                    window, opportunity=opportunity, source_id=source_id,
                    label_end_ns=int(market.timestamps[row + opportunity_contract["horizon"]].astype("datetime64[ns]").astype(np.int64)),
                    economic_contract={
                        **opportunity_contract,
                        "risk_dollars": environment.spec.per_trade_risk_dollars,
                        "point_value": environment.tick_values[ticker],
                        "round_trip_fee": environment.round_trip_fees[ticker],
                        "execution": "next_bar_open", "same_bar_collision": "adverse_first",
                    },
                )
            yield {"action": action_record, "market": market_record}
            emitted += 1
            if emitted >= maximum_examples:
                return
        action = behavior(observation, info)
        observation, _, terminated, truncated, info = environment.step(action)
        prefix.append(action)
        row = int(info["fill_index"])
        if terminated or truncated:
            return
    raise ValueError("collection episode incomplete within budget")
