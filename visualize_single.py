import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import sprl.environments
from sprl.distributions.kl_joint import KLJoint
from visualize_results import compute_rewards_and_successes, visualize_distribution_evolution, visualize_perf_or_suc

index_gate = {
    "experiment": 0,
    "idx": [0, 50, 150, 249,],
    "alphas": 0.3,
    "x_ticks": None,
    "y_ticks": None,
    "x_label": "Gate Position",
    "y_label": "Gate Width",
    "bounds_overwrite": None
}

FONTSIZE = 10


def main():
    file_str = sys.argv[-1]
    file = Path(file_str)

    data = pickle.load(file.open('rb'))
    policies = []
    distributions = []
    for i in range(0, len(data)):
        sub_policies = []
        sub_distributions = []
        for j in range(0, len(data[i][0])):
            if isinstance(data[i][0][j], KLJoint):
                sub_policies.append(data[i][0][j].policy)
                sub_distributions.append(data[i][0][j].distribution)
            else:
                sub_policies.append(data[i][0][j])
        policies.append(sub_policies)
        distributions.append(sub_distributions)

    env = sprl.environments.get(sys.argv[-2], cores=1)
    rs = compute_rewards_and_successes(env, policies)
    rs["distributions"] = distributions

    _, ax = plt.subplots(2, 1)

    visualize_perf_or_suc([rs["idx"]], [rs["successes"]], ["SPRL-SVGD"], False,
                          FONTSIZE, ["C0"], ax[0])

    visualize_distribution_evolution(
        env,
        index_gate["idx"],
        rs["distributions"][index_gate["experiment"]],
        index_gate["alphas"], FONTSIZE,
        index_gate["x_ticks"], index_gate["y_ticks"],
        index_gate["x_label"], index_gate["y_label"],
        index_gate["bounds_overwrite"],
        ax[1]
    )

    plt.show(block=True)


if __name__ == "__main__":
    main()