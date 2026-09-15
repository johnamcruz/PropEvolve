"""Public trade-assessment to executable-distribution contract."""
import numpy as np
import pytest

from propevolve.decision import Action


@pytest.mark.parametrize("action", list(Action))
@pytest.mark.parametrize("scores", [[-10., -10., -10.], [10., 10., 10.]])
def test_forced_legal_action_has_all_mass_and_is_selected(action, scores):
    from propevolve.reasoning_policy.staged_policy import legal_action_log_probs, select_legal_action
    assert np.exp(legal_action_log_probs(np.array(scores), (action,), xp=np)).tolist() == [1.]
    assert select_legal_action(scores, (action,)) == action


@pytest.mark.parametrize("scores,expected", [
    ([2., 2., -100.], Action.ENTER_LONG_1),
    ([2., -2., 100.], Action.ENTER_SHORT_1),
    ([-2., 2., 100.], Action.WAIT),
    ([-2., -2., -100.], Action.WAIT),
    ([0., 2., 0.], Action.WAIT),
    ([2., 0., 0.], Action.WAIT),
])
def test_flat_selection_respects_entry_direction_and_ties(scores, expected):
    from propevolve.reasoning_policy.staged_policy import select_legal_action, legal_action_log_probs
    actions = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    assert select_legal_action(scores, actions) == expected
    p = np.exp(legal_action_log_probs(np.array(scores), actions, xp=np))
    assert p.sum() == pytest.approx(1.)
    assert (p >= 0).all()


@pytest.mark.parametrize("score,expected", [(-2., Action.CLOSE), (0., Action.CLOSE), (2., Action.HOLD)])
def test_positioned_selection_ignores_entry_direction(score, expected):
    from propevolve.reasoning_policy.staged_policy import select_legal_action
    for unrelated in (-100., 100.):
        assert select_legal_action([unrelated, unrelated, score], (Action.HOLD, Action.CLOSE)) == expected


def test_separate_entry_and_direction_form_one_legal_distribution():
    from propevolve.reasoning_policy.staged_policy import legal_action_log_probs

    # ENTER=.8 and LONG|ENTER=.75: WAIT=.2, LONG=.6, SHORT=.2.
    result = legal_action_log_probs(
        np.array([np.log(4.), np.log(3.), 0.]),
        (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1), xp=np,
    )
    assert np.exp(result) == pytest.approx([.2, .6, .2])


@pytest.mark.parametrize("actions", [
    (), (Action.WAIT, Action.HOLD), (Action.WAIT, Action.WAIT),
])
def test_distribution_rejects_incoherent_legal_action_state(actions):
    from propevolve.reasoning_policy.staged_policy import legal_action_log_probs
    with pytest.raises(ValueError, match="legal actions"):
        legal_action_log_probs(np.zeros(3), actions, xp=np)


def test_management_and_hard_safety_masks_do_not_mix_decision_tasks():
    from propevolve.reasoning_policy.staged_policy import legal_action_log_probs
    hold_close = legal_action_log_probs(
        np.array([100., -100., np.log(3.)]), (Action.HOLD, Action.CLOSE), xp=np)
    assert np.exp(hold_close) == pytest.approx([.75, .25])
    forced_wait = legal_action_log_probs(
        np.array([100., 100., 100.]), (Action.WAIT,), xp=np)
    assert forced_wait.tolist() == [0.]


def test_mlx_distribution_keeps_gradients_and_matches_cpu_reference():
    mx = pytest.importorskip("mlx.core")
    from propevolve.reasoning_policy.staged_policy import legal_action_log_probs
    actions = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    logits = mx.array([np.log(4.), np.log(3.), 0.])
    loss, gradients = mx.value_and_grad(
        lambda x: -legal_action_log_probs(x, actions, xp=mx)[1])(logits)
    mx.eval(loss, gradients)
    assert float(loss) == pytest.approx(-np.log(.6), abs=1e-6)
    # Long likelihood = P(ENTER)*P(LONG|ENTER); management is inapplicable.
    assert gradients.tolist() == pytest.approx([-.2, -.25, 0.], abs=1e-6)


def test_binary_assessment_requires_exactly_three_scalar_log_odds():
    from propevolve.reasoning_policy.staged_policy import legal_action_log_probs
    with pytest.raises(ValueError, match="three scalar"):
        legal_action_log_probs(np.zeros((3, 2)), (Action.WAIT,), xp=np)
