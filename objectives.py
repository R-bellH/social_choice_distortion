import math
from collections import defaultdict

import numpy as np
from classes import Agent, Election, random_election
from social_choice_functions import random_dictatorship


# the sum objective  is the sum of agent costs for that particular alternative.
def sum_SC(x, d):
    """
    :param x: chosen alternative
    :param d: list of agents
    :return: sum of costs
    """
    return sum([agent.costs[x] for agent in d])


def med(x, d):
    """
    :param x: chosen alternative
    :param d: list of agents
    :return: median of costs
    """
    return np.median([agent.costs[x] for agent in d])


def probabilistic_cost(f, profile, objective, alpha):
    """
    :param f: probabilistic social choice function
    :param profile: list of agents
    :param objective: objective function
    :return: expected cost
    """
    sum = 0
    alternatives = f(profile, alpha)
    for alternative in alternatives.keys():
        p = alternatives[alternative]
        sum += p * objective(alternative, profile)
    return sum


def estimate_distortion(f, profile, objective, election, iterations=1000):
    """
    :param profile: list of agents
    :param f: probabilistic social choice function
    :param objective: objective function
    :return: estimation of the estimate_distortion over the objective function
    """
    max_dist = 0
    for i in range(iterations):
        fictitious_profile = []
        for agent in profile:
            fictitious_costs = agent.possible_costs(election.alpha)
            fictitious_agent = Agent(agent.id, fictitious_costs)
            fictitious_profile.append(fictitious_agent)
        objective_cost = probabilistic_cost(f, fictitious_profile, objective, election.alpha)
        true_optimal_cost = np.inf
        for alternative in election.alternatives:
            cost = objective(alternative, fictitious_profile)
            if cost < true_optimal_cost:
                true_optimal_cost = cost
        dist = objective_cost / true_optimal_cost
        if dist > max_dist:
            max_dist = dist
    return max_dist


def distortion(f, profile, objective, election):
    alpha = election.alpha
    m=len(profile[0].ordinal_preferences)
    winners = f(profile, alpha)
    epsilon = (m/2 - alpha) / (alpha + m/2)
    worst_distortion = 0
    for winner in winners.keys():  # winners is a dict {winner_id : probability that f choose winner}
        worst_profile = []
        for agent in profile:
            if agent.preference == winner:  # maximally indecisive reward for winner
                worst_costs = {candidate: m/2 + epsilon for candidate in election.candidates}
                worst_costs[winner] = m/2 - epsilon
            else:  # maximally decisive negative cost for the winner for the rest
                worst_costs = {candidate: 0.01 for candidate in election.candidates}
                worst_costs[winner] = m
            fictitious_agent = Agent(agent.id, worst_costs)
            worst_profile.append(fictitious_agent)
        realized_worst_cost = probabilistic_cost(f, worst_profile, objective, alpha)
        optimal_cost = min(objective(alternative, worst_profile) for alternative in election.alternatives)
        worst_distortion = max(worst_distortion, realized_worst_cost / optimal_cost)
    if round(worst_distortion,1) != 1.5:
        print(f.__name__)
        print(worst_distortion)
    return worst_distortion
