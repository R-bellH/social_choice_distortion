import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import objectives
from classes import random_election, random_election_with_alpha, Agent, Election
from social_choice_functions import random_dictatorship, plurality, borda
from social_choice_functions import alpha_Generalized_Proportional_to_Squares as alphaGPtS


def test_distortion(f, objective, election, agents):
    # distortion = objectives.estimate_distortion(f, agents, objective, election)
    distortion = objectives.distortion(f, agents, objective, election)
    # print(f"{objective.__name__} estimate_distortion for {f.__name__} is {estimate_distortion}")
    return distortion


def show_graph(distortion, alpha, objective, mean_distortion=None):
    sns.barplot(x=[f.__name__ for f in [random_dictatorship, plurality, borda]], y=distortion, color='r')
    if mean_distortion is not None:
        # add line to show mean estimate_distortion for each f
        plt.plot([f.__name__ for f in [random_dictatorship, plurality, borda]], mean_distortion, 'r--')
    # add in text the y value for each bar
    for i in range(3):
        plt.text(x=i - 0.1, y=distortion[i] + 0.05, s=round(distortion[i], 2), size=10)
    plt.title(f"distortion for {objective} objective with alpha = {round(alpha, 2)}")
    plt.show()


def double_graph(distortions, alpha, objectives, ):
    barWidth = 0.25
    r1 = np.arange(3)
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, distortions[0], color='b', width=barWidth, edgecolor='grey', label='sum_SC')
    plt.bar(r2, distortions[1], color='r', width=barWidth, edgecolor='grey', label='med')
    plt.xticks([r + barWidth / 2 for r in range(3)], [f.__name__ for f in [random_dictatorship, plurality, borda]])
    # add in text the y value for each bar
    for i in range(3):
        plt.text(x=r1[i] - 0.1, y=distortions[0][i] + 0.05, s=round(distortions[0][i], 2), size=10)
        plt.text(x=r2[i] - 0.1, y=distortions[1][i] + 0.05, s=round(distortions[1][i], 2), size=10)
    plt.title(f"distortions for {objectives} objective with alpha = {round(alpha, 2)}")
    # Create legend and place it a little left of the center
    plt.legend(loc='lower right')
    plt.show()


def simulate_elections(num_agents=20, num_candidates=2, alpha=0.5, verbose=False):
    election, agents = random_election_with_alpha(num_agents, num_candidates, alpha)
    distortions_SC = []
    for f in [random_dictatorship, plurality, borda]:
        distortions_SC.append(test_distortion(f, objectives.sum_SC, election, agents))
    distortions_med = []
    for f in [random_dictatorship, plurality, borda]:
        distortions_med.append(test_distortion(f, objectives.med, election, agents))
    if verbose:
        print("the agents are: ")
        for agent in agents:
            print(agent.ordinal_preferences)
        print("the election is: ")
        print(election.alternatives)
        alpha = election.get_best_alpha(agents)
        print("the alpha for the election is: ", alpha)
        show_graph(distortions_SC, alpha, "sum_SC")
        show_graph(distortions_med, alpha, "med")

    return distortions_SC, distortions_med


def static_example():
    num_agents = 2
    winners_n = 1
    num_candidates = 2
    alpha = 0.5
    epsilon = (1.0 - alpha) / (alpha + 1.0)
    winner_voters_costs = {0: 1.0+epsilon, 1: 1.0-epsilon}
    loser_voters_costs = {0: 0.0, 1: 2.0}
    agents_costs = []
    for i in range(winners_n):
        agents_costs.append(winner_voters_costs.copy())
    for i in range(num_agents - winners_n):
        agents_costs.append(loser_voters_costs.copy())
    agents = [Agent(i, costs) for i, costs in enumerate(agents_costs)]
    election = Election(list(range(num_candidates)))
    election.alpha = alpha
    distortions_SC = []
    for f in [random_dictatorship, plurality, borda, alphaGPtS]:
        dist = objectives.distortion(f, agents, objectives.sum_SC, election)
        distortions_SC.append(dist)
    print(distortions_SC)


## instead of the current graph show histogram of the the distortion for each f with a highlight for the worst case distortion. that will show how rare that is
def main():
    iterations = range(100)
    maximum_alpha = 0.5
    num_agents = 10
    num_candidates = 4
    sum_SC_distortions = []
    for _ in iterations:
        distortions_SC, _ = simulate_elections(num_agents, num_candidates, maximum_alpha)
        sum_SC_distortions.append(distortions_SC)
    sum_SC_distortions = np.array(sum_SC_distortions)
    # round distortion to 2 decimal points
    sum_SC_distortions = np.round(sum_SC_distortions, 2)

    deterministic_worst_case_bound = 2 * num_candidates - 1
    random_dictatorship_worst_case_upper_bound = 2.5 - 2 / num_agents
    random_dictatorship_worst_case_lower_bound = 1 + maximum_alpha
    alphaGPtS_worst_case_bound = 1 + maximum_alpha
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Plot histogram for the random dictatorship distortion
    sns.histplot(sum_SC_distortions[:, 0], kde=True, ax=axs[0, 0])
    axs[0, 0].axvline(random_dictatorship_worst_case_upper_bound, color='r', linestyle='dashed', linewidth=1)
    axs[0, 0].axvline(random_dictatorship_worst_case_lower_bound, color='r', linestyle='dashed', linewidth=1)
    axs[0, 0].set_title("sum_SC distortion histogram for random dictatorship")
    axs[0, 0].set_ylim([0, 80])  # Set y-axis limit

    # Plot histogram for the random plurality distortion
    sns.histplot(sum_SC_distortions[:, 1], kde=True, ax=axs[0, 1])
    axs[0, 1].axvline(deterministic_worst_case_bound, color='r', linestyle='dashed', linewidth=1)
    axs[0, 1].axvline(deterministic_worst_case_bound, color='r', linestyle='dashed', linewidth=1)
    axs[0, 1].set_title("sum_SC distortion histogram for plurality")
    axs[0, 1].set_ylim([0, 80])  # Set y-axis limit

    # Plot histogram for the borda distortion
    sns.histplot(sum_SC_distortions[:, 2], kde=True, ax=axs[0, 2])
    axs[0, 2].axvline(deterministic_worst_case_bound, color='r', linestyle='dashed', linewidth=1)
    axs[0, 2].axvline(deterministic_worst_case_bound, color='r', linestyle='dashed', linewidth=1)
    axs[0, 2].set_title("sum_SC distortion histogram for borda")
    axs[0, 2].set_ylim([0, 80])  # Set y-axis limit

    # Plot histogram for the alphaGPtS
    sns.histplot(sum_SC_distortions[:, 2], kde=True, ax=axs[1, 1])
    axs[1, 1].axvline(alphaGPtS_worst_case_bound, color='r', linestyle='dashed', linewidth=1)
    axs[1, 1].axvline(alphaGPtS_worst_case_bound, color='r', linestyle='dashed', linewidth=1)
    axs[1, 1].set_title("sum_SC distortion histogram for alpha Generalized Proportional to Squares")
    axs[1, 1].set_ylim([0, 80])  # Set y-axis limit

    plt.tight_layout()  # Adjust the padding between and around the subplots
    plt.show()


if __name__ == "__main__":
    main()
    # static_example()
    # iterations = range(100)
    # maximum_alpha = 0.5
    # num_agents = 10
    # num_candidates = 2
    # sum_SC_distortions = []
    # med_distortions = []
    # for _ in iterations:
    #     distortions_SC, distortions_med = main(num_agents, num_candidates, maximum_alpha)
    #     sum_SC_distortions.append(distortions_SC)
    # sum_SC_distortions = np.array(sum_SC_distortions)
    # med_distortions = np.array(med_distortions)
    # mean_SC_distortions = sum_SC_distortions.mean(axis=0)
    # mean_med_distortions = med_distortions.mean(axis=0)
    # # we want to show that at the worst case the estimate_distortion from the random dictatorship is the lowest and bound by 1+alpha
    # sum_SC_distortions = sum_SC_distortions.max(axis=0)
    # med_distortions = med_distortions.max(axis=0)
    # show_graph(sum_SC_distortions, maximum_alpha, "sum_SC")
    # # show_graph(med_distortions, maximum_alpha, "med", mean_med_distortions)
    # # double_graph([sum_SC_distortions, med_distortions], maximum_alpha, "sum_SC and med")
    #
