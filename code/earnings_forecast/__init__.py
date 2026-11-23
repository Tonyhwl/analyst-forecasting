"""Forecast quarterly EPS, revenue and operating margin from analyst consensus.

Modules, in the order data flows through them:

- ``config``: paths, evaluation periods and fixed model settings
- ``data``: S&P 500 rosters and Capital IQ release histories
- ``features``: inputs built only from earlier releases
- ``models``: boosted consensus corrections and logistic probabilities
- ``inference``: paired comparisons clustered by announcement quarter
- ``eps``, ``explain``, ``revenue_margin``, ``miss_risk``: one study each
- ``pipeline``: run every study and compare with the saved outputs
"""
