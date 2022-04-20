import numpy as np
from Graph import Graph, F
from uai_loader import load
import sys
from LL_dif import dif_sum

# data = np.loadtxt("data/dataset1/train-f-1.txt", skiprows=1, dtype='int')
data = np.loadtxt("sample_dataset.txt", skiprows=1, dtype='int')
# print(data)
#print(data.shape)
# print(data[0])

def get_cpt(factor, data):
    """

    :param factor: a CPT
    :param data: the samples
    :return:
    """
    factor.table = np.zeros_like(factor.table)
    CPT = factor.table
    rvs = factor.nb
    if len(rvs) == 0:
        print("We have no variables")
        return
    # get the child
    child = rvs[-1]
    num_rvs = len(rvs)
    num_parent = len(rvs) - 1
    for index, x in np.ndenumerate(factor.table):
        condition_Y = True
        if num_parent == 0:
            count_Y = data.shape[0]
        else:
            for i in range(num_parent):
                rv = rvs[i]
                value = index[i]
                condition_Y = condition_Y & (data[:, rv.name] == value)
                count_Y = np.sum(condition_Y)

        # if count_Y == 0:
        #     CPT[index] = 0
        #     continue

        condition = True
        for i in range(num_rvs):
            rv = rvs[i]
            value = index[i]
            condition = condition & (data[:, rv.name]==value)

        count_XY = np.sum(condition)
        #print("count XY: ", count_XY)
        #print("count_Y: ", count_Y)
        CPT[index] = (count_XY + 1) / (count_Y + 2)

    # print(CPT)

print("-------------")

rvs, fs = load('sample_bayes.uai')
true_rvs, true_fs = load('sample_bayes.uai')
# rvs, fs = load('data/dataset1/1.uai')
#print(fs)
# Learning the parameters
for i,f in fs.items():
    get_cpt(f, data)


# # Original paremeters
# for i, f in true_fs.items():
#     print(f.table)
# for i in range(5):
#     get_cpt(fs[i], data)
print("****************")
# def get_probs_from_CPTs(sample, fs):
#     probs = []
#
#     for i, f in fs.items():
#         cpt = f.table
#         rvs = f.nb
#         values = []
#         # get the index (variable assigment) of CPT
#         for rv in rvs:
#             value = sample[rv.name]
#             values.append(value)
#
#         values = tuple(values)
#         prob = cpt[values]
#         # avoid the probability of 0 or small value
#         if prob == 0 or prob < 1e-6:
#             prob = 1e-6
#
#         probs.append(prob)
#
#     #print("probs for this assignment: ", probs)
#     return probs
#
#
# def log_likelihood(probs):
#     result = 0
#     for p in probs:
#         result += np.log(p)
#     return result
#
#
# # probs = get_probs_from_CPTs([1,1,1], fs)
# # result = log_likelihood(probs)
# # print("log likelihood: ", result)
#
# def log_likelihood_per_sample(sample, fs):
#     probs = get_probs_from_CPTs(sample, fs)
#     result = log_likelihood(probs)
#     return result
# #
# # result = log_likelihood_per_sample([1,1,1], fs)
# # print("log likelihood from learned model: ", result)
# #
# # result = log_likelihood_per_sample([1,1,1], true_fs)
# # print("log likelihood from true model: ", result)
#
# def diff_per_sample(sample, fs, true_fs):
#     LL_learned = log_likelihood_per_sample(sample, fs)
#     LL_true = log_likelihood_per_sample(sample, true_fs)
#     return np.abs(LL_learned - LL_true)
# # print("difference: ", diff_per_sample([1,1,1], fs, true_fs))
#
# def dif_sum(test_file, fs, true_fs):
#     # samples = np.loadtxt("data/dataset1/test.txt", skiprows=1, dtype='int')
#     samples = np.loadtxt(test_file, skiprows=1, dtype='int')
#     total = 0
#     for i in range(samples.shape[0]):
#         dif_sample = diff_per_sample(samples[i], fs, true_fs)
#         print(f"{i}th dif: {dif_sample}")
#         total += dif_sample
#
#     print("log likelihood difference = : ", total)
#     return total

dif_sum("sample_test.txt", fs, true_fs)


def main():
    uai_file = sys.argv[1]
    """to be completed for task 1"""