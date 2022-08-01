from featLearnBan.policies.SA_FeatLearnBan import SA_FeatLearnBan
from featLearnBan.policies.Random import Random
from featLearnBan.policies.Oful import Oful
from featLearnBan.policies.Lasso import Lasso
from featLearnBan.policies.Greedy import Greedy
from featLearnBan.environment import MTL_ICML as MTL
from featLearnBan.Evaluation import Evaluation

from numpy import arange, cumsum, load, log, real, save, zeros
from optparse import OptionParser
import matplotlib.pyplot as plt
from matplotlib import rc
import os.path


# ===
# RUNNING PARAMETERS
# ===

VERBOSE = True

parser = OptionParser("Usage: %prog [options]", version="%prog 1.0")
parser.add_option('-d', dest = 'd', default = "20", type = 'int', help = "Number of features")
parser.add_option('-r', dest = 'r', default = "5", type = 'int', help = "Rank value")
parser.add_option('-K', dest = 'K', default = "10", type = 'int', help = "Number of arms")
parser.add_option('-T', dest = 'T', default = "40", type = 'int', help = "Number of rounds")
parser.add_option('-N', dest = 'N', default = "10", type = 'int', help = "Number of tasks")
parser.add_option('--nrep', dest = 'nrep', default = "1", type = 'int', help = "Number Repetitions")


(opts, args) = parser.parse_args()
K = opts.K
T = opts.T
N_TASK = opts.N
d = opts.d
r = opts.r
nrep = opts.nrep

noisy_rewards = True
variance = 1.

assert K > 0, "Not consistent arms number parameter"
assert nrep > 0, "Not consistent number of repetitions"
assert d >= r, "Not consistent sparsity-features relation"
assert T >= r, "Not consistent sparsity-features relation"
assert T > 0, "Not consistent horizon parameter"
assert N_TASK > 0, "Not consistent number of tasks"

if VERBOSE:
    print("RUN Horizon {}, Tasks {}, Arms {}, dimensions {}, rank {}".format(T, N_TASK, K, d, r))

# ===
# INITIALIZATION
# ===
# Policies
policies = {}
policies_name = []

reg = [.1, 1, 10, 100]

# Random Policy
policies["Trace-Norm Bandit"] = [SA_FeatLearnBan(N_TASK, T, d ,K, reg[0])] 
policies_name.append("Trace-Norm Bandit")

assert N_TASK >= nrep, "This is required to have a consistent seed"

# Plot parameters
colors = ['r', 'g', 'c', 'b', 'y', 'm', '#ae34eb', '#eb348f', '#000000']
COLORS = {p_name: colors[i] for i, p_name in enumerate(policies_name)}
markers = ["o", "^", "v", "<", ">", ",", "h", "x", "+"]
MARKERS = {p_name: markers[i] for i, p_name in enumerate(policies_name)}
linestyle = ["-", "-", "--", "-.", ":", "--", "-.", ":", "-"]
LINESTYLE = {p_name: linestyle[i] for i, p_name in enumerate(policies_name)}

# ===
# RUN OVER POLICIES
# ===
What_file_str = "What_tasks{}_hor{}_d{}_r{}.npy".format(N_TASK, T, d, r)
results = []
test_results = []

if not os.path.exists(What_file_str):
    print("What to be computed")
    for p_name, p in policies.items():
        print("\n=======NEW RUN=======")
        print("===POLICY {}===".format(p_name))
        # Here each policy p is a list of N_TASK policies except for SA_FeatLearnBan
        mtl = MTL(N_TASK, T, K, d, r, variance, p, p_name, noisy_rewards)
        evaluation = Evaluation(mtl, T, p_name, nrep, N_TASK)
        results.append(evaluation.getResults())  # Result Class: (Policy Name, Mean Rewards, Std Rewards)
        if p_name == "Trace-Norm Bandit":
            What = p[0].get_What()
            with open(What_file_str, "wb") as f:
                save(f, What)
            print("What shape {}".format(What.shape))
else:
    print("Loading What")
    with open(What_file_str, "rb") as f:
        What = load(f)

What = What.reshape((N_TASK, d))
from numpy.linalg import matrix_rank, qr
rhat = matrix_rank(What)
Q, R = qr(What)
Bhat = real(R[:rhat, :d].T)
print("Shape of What {} Bhat {}, rhat {}".format(What.shape, Bhat.shape, rhat))
pol = {}
pol_name = ["Lasso", "Greedy", "OFUL"]
pol["Lasso"] = [Lasso(T, d, K, 1)]
pol["Greedy"] = [Greedy(T, d , K, Bhat)]
pol["OFUL"] = [Oful(T, d, K, 1)]
results = []

for p_n, p in pol.items():
    print("RUN {}".format(p_n))
    mab = MTL(1, T, K, d, r, variance, p, p_n, noisy_rewards)
    evaluation = Evaluation(mab, T, p_n, nrep, 1) # env, horizon, policy name, nrep, nbTasks
    results.append(evaluation.getResults())

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
round_indexes = arange(T)

p_index = 0 
for name, avg, std in results:
    cumsumavg = cumsum(avg)
    print("Pol {} got {}".format(name, cumsumavg[-1]))
    plt.fill_between(round_indexes, cumsumavg - std/2, cumsumavg + std/2, alpha = 0.5, color = colors[p_index])
    plt.plot(round_indexes, cumsumavg, color = colors[p_index], marker = markers[p_index], label = name)
    p_index += 1
ax.set_ylabel("Cumulative Rewards")
ax.set_xlabel("Round across Tasks")
ax.set_title("LTL with {} past Tasks".format(N_TASK))
ax.yaxis.grid(True)
plt.legend()
plt.savefig("output/NEURIPS_d{}_rhat{}_T{}_rep{}_N{}_K{}.png".format(d, rhat, T, nrep, N_TASK, K))
