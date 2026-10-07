"""deskmind#63 part 2, brain's side: a click on a commit control goes to the strong tier like a DONE, and a strong-tier
answer that overrules a fairly sure fast DONE must be sure itself (E1, 10-07)."""
import unittest

from deskmind_brain.router import decide, is_commit_click, route


def click(label, role="button", p=0.961, list_form=False):
    crit = ([{"key": "1", "element": f"[1] {label}", "role": role}] if list_form
            else {"1": {"element": f"[1] {label}", "role": role}})
    body = {"questions": {"operation": {}, "click_target": {"criteria": crit}}, "state": {}}
    answers = {"operation": {"choice": "CLICK", "probabilities": {"CLICK": p, "DONE": 1 - p}},
               "click_target": {"choice": "1", "probabilities": {"1": 0.99}}}
    return body, answers


class CommitClicks(unittest.TestCase):
    def test_commit_clicks_escalate_even_when_sure(self):
        for label in ("保存", "提交", "发送", "删除", "Save", "Submit", "Send", "Delete Message", "立即支付"):
            for list_form in (False, True):
                body, answers = click(label, list_form=list_form)
                self.assertEqual(decide(body, answers, 0.94, 0.8)[:2], (True, "risky_commit"), (label, list_form))

    def test_look_alikes_and_non_actions_do_not(self):
        for label, role in (("删除线", "button"), ("保存位置", "button"), ("Save As…", "menuItem"), ("Saved", "button"),
                            ("发送时间", "heading"), ("保存", "checkbox"), ("Submissions", "button"), ("打开", "button"),
                            ("Send me a copy of every message I write", "button")):
            body, answers = click(label, role)
            self.assertFalse(is_commit_click(body, answers), (label, role))
            self.assertEqual(decide(body, answers, 0.94, 0.8)[1], "fast_ok", label)


class DoneOverride(unittest.TestCase):
    BODY = {"questions": {"operation": {}, "click_target": {"criteria": {"1": {"element": "[1] 收件箱", "role": "row"}}}},
            "state": {}}

    def fast_done(self, p):
        return {"operation": {"choice": "DONE", "probabilities": {"DONE": p, "CLICK": 1 - p}}}

    def strong_click(self, p):
        return lambda body: {"operation": {"choice": "CLICK", "probabilities": {"CLICK": p, "DONE": 1 - p}},
                             "click_target": {"choice": "1", "probabilities": {"1": 0.99}}}

    def test_an_unsure_override_of_a_sure_done_keeps_the_done(self):
        answers, rec = route(self.BODY, self.fast_done(0.97), self.strong_click(0.82))
        self.assertEqual(answers["operation"]["choice"], "DONE")
        self.assertEqual((rec["by"], rec["reason"]), ("fast", "done_kept_low_override"))

    def test_a_sure_override_wins(self):
        answers, rec = route(self.BODY, self.fast_done(0.97), self.strong_click(0.98))
        self.assertEqual(answers["operation"]["choice"], "CLICK")
        self.assertEqual(rec["by"], "strong")

    def test_an_unsure_fast_done_is_not_kept(self):
        answers, rec = route(self.BODY, self.fast_done(0.6), self.strong_click(0.82))
        self.assertEqual(answers["operation"]["choice"], "CLICK")
        self.assertEqual(rec["reason"], "risky_DONE")


if __name__ == "__main__":
    unittest.main()
