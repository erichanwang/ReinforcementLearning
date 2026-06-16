# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through value iteration.
# The discount grid is:
#
#    _    _    _    _    _       (y=4)
#    _    #    _    _    _       (y=3)
#    _    #    1    #   10       (y=2)  <- exits
#    S    _    _    _    _       (y=1)
#  -10  -10  -10  -10  -10      (y=0)  <- exits
#
# Close exit: reward 1 at (2,2); Far exit: reward 10 at (4,2)


def question2():
    # Cross the bridge: reduce noise to 0 so it is safe to walk the narrow bridge.
    # High discount keeps the far reward (10) attractive.
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise


def question3a():
    # Prefer close exit (1), take risky short path (adjacent to -10 row).
    # Low discount makes the nearby exit attractive; zero noise makes the risky path safe.
    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward


def question3b():
    # Prefer close exit (1), take safe long path (around the top).
    # Low discount still favours the close exit; nonzero noise makes the risky path dangerous.
    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward


def question3c():
    # Prefer far exit (10), take risky short path.
    # High discount keeps the far reward worthwhile; zero noise makes the risky path safe.
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward


def question3d():
    # Prefer far exit (10), take safe long path.
    # High discount keeps the far reward worthwhile; nonzero noise makes the risky path dangerous.
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward


def question3e():
    # Avoid both exits: a very high living reward makes staying alive perpetually optimal.
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 100.0
    return answerDiscount, answerNoise, answerLivingReward


def question8():
    # It is not possible to simultaneously have the agent prefer the close exit
    # AND risk the bottom row in the BridgeGrid under any noise/discount combination.
    return 'not possible'
