"""A DONE right after an action whose effect could not be confirmed escalates with its own reason."""
import unittest

from deskmind_brain.router import decide


class UnverifiedLast(unittest.TestCase):
    ANSWERS = {"operation": {"choice": "DONE", "probabilities": {"DONE": 0.97, "CLICK": 0.03}}}

    def test_reason_when_the_last_action_was_unconfirmed(self):
        for eff in ("unverifiable", "suspected_noop"):
            body = {"questions": {"operation": {}}, "state": {"last_effect": eff}}
            self.assertEqual(decide(body, self.ANSWERS, 0.94, 0.8)[:2], (True, "unverified_last"))

    def test_plain_done_is_still_risky(self):
        for state in ({}, {"last_effect": "confirmed"}):
            body = {"questions": {"operation": {}}, "state": state}
            self.assertEqual(decide(body, self.ANSWERS, 0.94, 0.8)[:2], (True, "risky_DONE"))


if __name__ == "__main__":
    unittest.main()
