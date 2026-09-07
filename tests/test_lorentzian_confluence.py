"""LC uses past Expansion examples, not Trend or future economic labels."""
from types import SimpleNamespace

import numpy as np

from propevolve.teachers.lorentzian import LorentzianTargets


def test_lc_can_use_existing_expansion_history_without_new_indicators():
    class ExpansionHistory:
        def target(self, ticker, row):
            current=[.9,.8,.1,.2]
            older=[.1,.2,.9,.8] if row != 1 else [.9,.8,.1,.2]
            return np.array(current+older)
    source=LorentzianTargets.build(
        {'NQ':SimpleNamespace(close=np.array([100.,110.,90.]))},
        ExpansionHistory(), dict(neighbors=1,history_size=4,label_horizon=1,
                                 minimum_move=0.,minimum_vote=0.),
    )
    # Same current score on all rows; prior sequence selects the bullish example.
    np.testing.assert_array_equal(source.target('NQ',2),[1,0,1,1])


def test_economic_targets_cannot_use_unmatured_outcomes():
    class Expansion:
        def target(self, ticker, row):
            return np.array([.9, .8, .1, .2])
    markets={'NQ':SimpleNamespace(close=np.array([100.,110.,120.,130.]))}
    settings=dict(neighbors=1,history_size=4,label_horizon=2,
                  minimum_move=0.,minimum_vote=0.,label_source='economic')
    a=LorentzianTargets.build(markets,Expansion(),settings,
                             economic_labels={'NQ':np.array([-1,1,1,1])})
    b=LorentzianTargets.build(markets,Expansion(),settings,
                             economic_labels={'NQ':np.array([-1,-1,-1,-1])})
    for row in range(3):
        np.testing.assert_array_equal(a.target('NQ',row),b.target('NQ',row))
    np.testing.assert_array_equal(a.target('NQ',2),[0,1,1,1])


def test_lc_targets_are_causal_directional_confluence_not_an_entry_filter():
    class Expansion:
        def target(self, ticker, row):
            return np.array([0.9, 0.8, 0.1, 0.2], dtype=np.float32)

    settings = dict(neighbors=1, history_size=4, label_horizon=1,
                    minimum_move=0.0, minimum_vote=0.0)
    market = SimpleNamespace(close=np.array([100., 110., 90.]))
    source = LorentzianTargets.build({'NQ': market}, Expansion(), settings)
    np.testing.assert_array_equal(source.target('NQ', 0), [0, 0, 1, 1])
    np.testing.assert_array_equal(source.target('NQ', 1), [1, 0, 1, 1])
    # Deterministic equal-distance ties prefer the older example.
    np.testing.assert_array_equal(source.target('NQ', 2), [1, 0, 1, 1])
    changed = LorentzianTargets.build(
        {'NQ': SimpleNamespace(close=np.array([100., 110., 1000.]))},
        Expansion(), settings,
    )
    np.testing.assert_array_equal(source.target('NQ', 1), changed.target('NQ', 1))
    # Random replay access cannot update LC or allow future examples to vote.
    source.target('NQ', 2)
    np.testing.assert_array_equal(source.target('NQ', 0), [0, 0, 1, 1])


def test_missing_expansion_does_not_remove_other_teacher_targets_or_bridge_gaps():
    class Expansion:
        def target(self, ticker, row):
            return None if row == 1 else np.array([.9, .8, .1, .2])
    source = LorentzianTargets.build(
        {'NQ': SimpleNamespace(close=np.array([100., 110., 90., 80.]))},
        Expansion(), dict(neighbors=1, history_size=4, label_horizon=1,
                          minimum_move=0., minimum_vote=0.),
    )
    np.testing.assert_array_equal(source.target('NQ', 1), [0, 0, 1, 1])
    np.testing.assert_array_equal(source.target('NQ', 2), [0, 0, 1, 1])
    np.testing.assert_array_equal(source.target('NQ', 3), [0, 1, 1, 1])


def test_lc_evidence_drives_existing_soft_loss_without_overriding_economic_truth():
    import torch
    from propevolve.trend_start_confluence import trend_start_confluence_rank_loss

    class Expansion:
        def target(self, ticker, row):
            return np.array([.9, .8, .1, .2])

    source = LorentzianTargets.build({
        'UP': SimpleNamespace(close=np.array([100., 110.])),
        'DOWN': SimpleNamespace(close=np.array([100., 90.])),
    }, Expansion(), dict(neighbors=1, history_size=4, label_horizon=1,
                         minimum_move=0., minimum_vote=0.))
    up, down = source.target('UP', 1), source.target('DOWN', 1)
    evidence = torch.tensor(np.stack([up, down, down, up, down, up, up, down]))
    q = torch.zeros((8, 3), requires_grad=True)
    loss = trend_start_confluence_rank_loss(
        q, action_targets=torch.tensor([1, 2, 0, 0, 1, 2, 0, 0]),
        economic_sides=torch.tensor([1, 2, 1, 2, 1, 2, 1, 2]),
        economic_wins=torch.tensor([True, True, False, False, True, True, False, False]),
        directional_scores=evidence[:, :2]*evidence[:, 2:], margin=.25,
    )
    loss.loss.backward()
    assert q.grad[0, 1] < 0 and q.grad[0, 0] > 0
    assert q.grad[1, 2] < 0 and q.grad[1, 0] > 0
    assert q.grad[2, 0] < 0 and q.grad[2, 1] > 0
    assert q.grad[3, 0] < 0 and q.grad[3, 2] > 0
    # Contrary winners and LC-aligned failures retain their economic labels;
    # LC alone cannot penalize a winner or promote a failure to ENTER.
    assert torch.equal(q.grad[4:], torch.zeros((4, 3)))
