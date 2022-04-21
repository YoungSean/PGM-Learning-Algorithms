import numpy as np
from Graph import Graph, F
from uai_loader import load
import sys
from LL_dif import dif_sum
import time
import os

# data = np.loadtxt("data/dataset1/train-f-1.txt", skiprows=1, dtype='int')
# data = np.loadtxt("sample_dataset.txt", skiprows=1, dtype='int')
# print(data)
#print(data.shape)
# print(data[0])

def get_cpt_mle(factor, data):
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

#print("-------------")


#
# rvs, fs = load('sample_bayes.uai')
# true_rvs, true_fs = load('sample_bayes.uai')
# # rvs, fs = load('data/dataset1/1.uai')
# #print(fs)
# # Learning the parameters
# for i,f in fs.items():
#     get_cpt_mle(f, data)


def train_mle(uai_file, train_data):
    print(f"start training with {os.path.basename(train_data)} and BayesNet {os.path.basename(uai_file)}")
    start_time = time.time()
    data = np.loadtxt(train_data, skiprows=1, dtype='int')
    rvs, fs = load(uai_file)
    for i, f in fs.items():
        get_cpt_mle(f, data)

    average_time = (time.time() - start_time)
    average_time = str(round(average_time, 3))
    print(f"The training time is: {average_time} s")

    return rvs, fs

def get_original_model(uai_file):
    #print(f"Loading the original network: {os.path.basename(uai_file)}")
    true_rvs, true_fs = load(uai_file)
    #print("Finish loading.")
    return true_rvs, true_fs

def learn_mle(uai_file, train_data, test_data):
    print("-------------MLE-------------------")
    rvs, fs = train_mle(uai_file, train_data)
    true_rvs, true_fs = get_original_model(uai_file)
    print("Evaluating the log likelihood difference...")
    start_time = time.time()
    dif = dif_sum(test_data, fs, true_fs)
    average_time = (time.time() - start_time)
    average_time = str(round(average_time, 3))
    print(f"The evaluating time is: {average_time} s.")
    print("------------------------")
    return dif


#dif = learn_mle('sample_bayes.uai', "sample_dataset.txt", "sample_test.txt")
# dif = learn_mle('data/dataset1/1.uai', "data/dataset1/train-f-1.txt", "data/dataset1/test.txt")

mle_result = "mle_result.txt"

uai_files = ['data/dataset1/1.uai', 'data/dataset2/2.uai', 'data/dataset3/3.uai']
mle_train_files = ["train-f-1.txt", "train-f-2.txt", "train-f-3.txt", "train-f-4.txt"]

def train_dataset(uai_file, train_dir, test_data, result_file):
    for train_file in mle_train_files:
        train_data = train_dir + "/" + train_file

        dif = learn_mle(uai_file, train_data, test_data)
        with open(result_file, 'a') as f:
            dif = str(round(dif, 4))
            f.write(os.path.basename(uai_file) + "\t" + train_file + "\t")
            f.write("LLDiff = " + dif + "\n")

def main():
    train_dataset('data/dataset1/1.uai', 'data/dataset1', 'data/dataset1/test.txt', mle_result)
    train_dataset('data/dataset2/2.uai', 'data/dataset2', 'data/dataset2/test.txt', mle_result)
    train_dataset('data/dataset3/3.uai', 'data/dataset3', 'data/dataset3/test.txt', mle_result)

if __name__ == '__main__':
    main()
# with open(mle_result, "a") as f:
#     dif = learn_mle('sample_bayes.uai', "sample_dataset.txt", "sample_test.txt")
#     dif = str(round(dif, 4))
#     f.write("LLDiff = " + dif + "\n")


# # Original paremeters
# for i, f in true_fs.items():
#     print(f.table)
# for i in range(5):
#     get_cpt(fs[i], data)
# print("****************")
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

# dif_sum("sample_test.txt", fs, true_fs)


