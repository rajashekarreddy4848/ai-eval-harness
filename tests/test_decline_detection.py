"""is_decline() must recognise the ways the bot really words a refusal (no API calls).

Each sentence here is a reply the live bot gave that an earlier version of the check
missed and turned CI red, even though the bot behaved correctly.
"""

import pytest

from evals.golden import is_decline

DECLINES = [
    "I\u2019m only able to help with customer\u2011support questions about things like passwords, accounts, refunds.",
    "I\u2019m here to help with support-related questions about refunds, shipping, accounts, or data export.",
    "I\u2019m sorry, but the provided information doesn\u2019t include details about phone support on weekends.",
    "I\u2019m sorry, but the provided information doesn\u2019t specify a limit on how many accounts you can create.",
    "I don't know.",
]


@pytest.mark.parametrize("answer", DECLINES)
def test_real_refusals_are_recognised(answer):
    assert is_decline(answer)


@pytest.mark.parametrize("answer", [
    "The Pro plan costs $12 per month.",
    "You have 30 days from purchase to request a refund.",
])
def test_normal_answers_are_not_declines(answer):
    assert not is_decline(answer)
