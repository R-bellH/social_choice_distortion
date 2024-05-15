from collections import defaultdict

import numpy as np


## Deterministic social choice functions

def plurality(profile, alpha):
    """
    :param profile: list of agents
    """
    candidates = defaultdict(int)
    for agent in profile:
        candidates[agent.preference] += 1
    candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    winner = {candidates[0][0]: 1}
    return winner


def borda(profile, alpha):
    """
    :param profile: list of agents
    """
    n = len(profile[0].ordinal_preferences)
    candidates = defaultdict(int)
    for agent in profile:
        for i in range(n):
            candidates[agent.ordinal_preferences[i]] += n - i
    candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    winner = {candidates[0][0]: 1}
    return winner


## Random social choice functions
# probabilistic social choice function returns a dictionary of preferences and the probability that the function chooses each preference

def random_dictatorship(profile, alpha):
    """
    :param profile: list of agents
    """
    n = len(profile)
    preferences = defaultdict(int)
    for agent in profile:
        preferences[agent.preference] += 1
    for preference in preferences.keys():
        preferences[preference] /= n
    return preferences

# alpha Generalized Proportional to Squares from page 14 in the paper
def alpha_Generalized_Proportional_to_Squares(profile, alpha):  # currently implemented only for m=2
    m = len(profile[0].ordinal_preferences)
    if m != 2:
        raise ValueError("alphaGPtS Currently implemented only for m=2")
    preferences = defaultdict(int)
    X = profile[0].ordinal_preferences[0]
    X_votes = len([agent for agent in profile if agent.preference == X])
    Y = profile[0].ordinal_preferences[1]
    Y_votes = len([agent for agent in profile if agent.preference == Y])

    preferences[Y] = sigmoid_try(proportional_to_squares(X_votes, Y_votes, alpha))
    if preferences[Y] > 1:
        print(preferences[Y])
        print(f"Y: {Y_votes}")
    if preferences[Y] <0:
        print(preferences[Y])
        print(f"Y: {Y_votes}")
    # preferences[X] = proportional_to_squares(Y_votes, X_votes, alpha)
    preferences[X] = 1 - preferences[Y]
    return preferences

def proportional_to_squares(x, y, alpha):
    return ((((1 + alpha) * y ** 2) - ((1 - alpha) * x * y)) /
            (((1 + alpha) * (x ** 2 + y ** 2)) - (2 * (1 - alpha) * x * y)))

def sigmoid_try(x):
    return 1/(1+np.exp(-x))
# def euclidean_1(profile):
#     n=len(profile)
#     candidates=profile[0].ordinal_preferences
#     c_minus=profile[0].preference
#     c_plus=profile[n-1].preference
#     for c in candidates:
#         if c>
