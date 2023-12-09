from ValueIteration import ValueIteration
from plot import plot_line
import numpy as np
from Simulator import Simulator
import random

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#Graph 1
# vi = ValueIteration(num_grid_cols=3, num_grid_rows=3, num_crates=3, obstacles={'o1': (1, 1)}, gamma=0.7)
# vi.initialise_value_matrix()
# delta_arr, count = vi.find_optimal()
# #print(delta_arr)
# plt.plot(range(len(delta_arr)), delta_arr)
# title = 'Value Iteration - Max Norm Trend'
# plt.title(title)
# plt.xlabel("Iteration Number")
# plt.ylabel("'Max norm between v(i+1) and v(i)")
# plt.legend()
# plt.savefig(title.replace(' ', '_') + '.png')
# plt.show()

#Graph 2
# gamma_arr = np.arange(0.1,1.1, 0.1)
# for gamma in gamma_arr:
#     vi = ValueIteration(num_grid_cols=3, num_grid_rows=3, num_crates=3, obstacles={'o1': (1, 1)}, gamma=gamma)
#     vi.initialise_value_matrix()
#     delta_arr, count = vi.find_optimal()
#     #print(delta_arr)
#     plt.plot(range(len(delta_arr)), delta_arr, label="gamma=" + format(gamma, ".1f"))
# title = 'Value Iteration - Max Norm Trend over gammas'
# plt.title(title)
# plt.xlabel("Iteration Number")
# plt.ylabel("'Max norm between v(i+1) and v(i)")
# plt.legend()
# plt.savefig(title.replace(' ', '_') + '.png')
# plt.show()


num_iterations = 10
gamma_arr = np.arange(0.1, 1.1, 0.1)
average_return_arr = []

vi = ValueIteration(num_grid_cols=3, num_grid_rows=3, num_crates=3, obstacles={'o1': (1, 1)})
vi.initialise_value_matrix()
state_space = vi.states

# Generate a list of random starting states for all iterations
random_starting_states = random.sample(state_space, num_iterations)

for gamma in gamma_arr:
    print(f"Gamma: {gamma}")
    return_sum = 0.0

    for idx in range(num_iterations):
        vi = ValueIteration(num_grid_cols=3, num_grid_rows=3, num_crates=3, obstacles={'o1': (1, 1)}, gamma=gamma)
        vi.initialise_value_matrix()

        starting_state = random_starting_states[idx]  # Use the predetermined random starting state

        optimal_policy = vi.get_optimal_policy()
        vi.slidingMap.updateStates(starting_state)
        sim = Simulator(vi.slidingMap)
        ret = sim.simulate(optimal_policy, 20)
        return_sum += ret
        del sim

    average_return = return_sum / num_iterations
    average_return_arr.append(average_return)

# Plotting average return for different gamma values
plt.plot(gamma_arr, average_return_arr)
title = 'Value Iteration - Testing Returns of optimal policy over different Gamma'
plt.title(title)
plt.xlabel("Gamma")
plt.ylabel("Optimal Return")
plt.legend()
plt.savefig(title.replace(' ', '_') + '.png')
plt.show()