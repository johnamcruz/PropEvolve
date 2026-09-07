"""Direction helper's public closed-bar flow, independent of any run config."""
import unittest

from propevolve.lorentzian_direction import LorentzianDirection


class DirectionFlowTests(unittest.TestCase):
    def helper(self, **overrides):
        settings = dict(neighbors=1, history_size=4, label_horizon=1,
                        minimum_move=0., minimum_vote=0.)
        return LorentzianDirection(**(settings | overrides))

    def test_forward_labels_mature_before_they_can_vote(self):
        helper = LorentzianDirection(neighbors=1, history_size=4,
            label_horizon=2, minimum_move=0., minimum_vote=0.)
        self.assertEqual(helper.update(0, (0.,), 100.).direction, 'uncertain')
        self.assertEqual(helper.update(1, (1.,), 90.).direction, 'uncertain')
        # First example describes the forward move 100 -> 110, not 100 -> 90.
        result = helper.update(2, (0.,), 110.)
        self.assertEqual(result.direction, 'long')
        self.assertEqual(result.vote, 1.)
        self.assertEqual(result.neighbors_used, 1)

    def test_short_and_uncertain_are_not_forced_long(self):
        helper = self.helper()
        helper.update(0, (0.,), 100.)
        self.assertEqual(helper.update(1, (0.,), 90.).direction, 'short')
        neutral = self.helper(minimum_move=.02)
        neutral.update(0, (0.,), 100.)
        self.assertEqual(neutral.update(1, (0.,), 101.).direction, 'uncertain')
        tie = self.helper(neighbors=2)
        tie.update(0, (0.,), 100.)
        tie.update(1, (0.,), 110.)
        self.assertEqual(tie.update(2, (0.,), 100.).vote, 0.)
        self.assertEqual(tie.update(3, (0.,), 100.).direction, 'uncertain')

    def test_actual_nearest_example_supplies_direction(self):
        helper = self.helper()
        helper.update(0, (0.,), 100.)       # This example becomes bullish.
        helper.update(1, (10.,), 110.)      # This one becomes bearish.
        self.assertEqual(helper.update(2, (10.,), 100.).direction, 'short')

    def test_future_suffix_does_not_change_historical_evidence(self):
        def run(prices):
            helper = self.helper()
            return [helper.update(i, (float(i % 2),), price) for i,price in enumerate(prices)]
        prefix = [100.,110.,105.,115.]
        self.assertEqual(run(prefix), run(prefix+[5.,900.])[:len(prefix)])

    def test_memory_is_bounded_and_rejected_input_does_not_advance(self):
        helper = self.helper(history_size=3,label_horizon=2)
        for i in range(20):helper.update(i,(float(i),),100.+i)
        self.assertEqual((helper.history_count,helper.pending_count),(3,2))
        for bar,features,price in [(20,(float('nan'),),100.),(20,(0.,1.),100.),
                                   (21,(0.,),100.),(20,(0.,),-1.)]:
            with self.assertRaises(ValueError):helper.update(bar,features,price)
            self.assertEqual((helper.history_count,helper.pending_count),(3,2))
        helper.update(20,(1.,),120.)

    def test_config_validation_and_strict_vote_threshold(self):
        for values in [dict(neighbors=0),dict(history_size=0),dict(label_horizon=0),
                       dict(neighbors=5),dict(minimum_vote=float('nan'))]:
            with self.assertRaises(ValueError):self.helper(**values)
        helper=self.helper(minimum_vote=1.)
        helper.update(0,(0.,),100.)
        self.assertEqual(helper.update(1,(0.,),110.).direction,'uncertain')

    def test_economic_label_must_be_mature_and_overrides_close_direction(self):
        helper = self.helper(label_source='economic', label_horizon=2)
        helper.update(0, (0.,), 100.)
        with self.assertRaises(ValueError):
            helper.update(1, (1.,), 110., matured_label=-1)
        helper.update(1, (1.,), 110.)
        with self.assertRaises(ValueError):
            helper.update(2, (0.,), 120.)
        with self.assertRaises(ValueError):
            helper.update(2, (0.,), 120., matured_label=2)
        # Price rose, but actual stop-first economics say Short was the winner.
        result = helper.update(2, (0.,), 120., matured_label=-1)
        self.assertEqual(result.direction, 'short')
        self.assertEqual(helper.history_count, 1)


if __name__ == '__main__': unittest.main()
