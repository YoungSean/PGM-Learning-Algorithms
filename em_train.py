import os
import time

import numpy as np
from collections import deque
from uai_loader import load
from LL_dif import log_likelihood_per_sample, dif_sum
from train import get_original_model

# get n numbers from 0 to 1. They sum to 1.
# used for parameter initialization
def get_prob(n):
    x = np.random.uniform(size=n)
    # softmax
    return np.exp(x)/sum(np.exp(x))


def line_to_cases(line):
    root = line.split()
    cases = []
    my_queue = deque([root])
    while my_queue:
        current = my_queue.popleft()
        for i, v in enumerate(current):
            if v == '?':
                current[i] = 0
                my_queue.append(list(current))
                current[i] = 1
                my_queue.append(list(current))
                # restore the current element to avoid appending into cases
                current[i] = '?'
                break
        if not '?' in current:
            current = [int(i) for i in current]
            cases.append(current)

    return np.array(cases), len(cases)

# line = '0 ? 1 ? ?'
#
# cases = line_to_cases(line)
# print(cases)

def get_initial_cases(train_file):
    size_missing = []
    with open(train_file, 'r') as f:
        first_line = f.readline()
        num_vr, num_rows = first_line.split()
        num_rows = int(num_rows)
        first_case = f.readline()
        cases,size_cases = line_to_cases(first_case)
        size_missing.append(size_cases)
        num_cases = cases.shape[0]
        #print("num_cases: ", num_cases)
        probs = get_prob(num_cases)
        #print('first case probs: ', probs)
        #print('first case dim: ', cases.shape)

        for i in range(num_rows - 1):
            # print(f"{i+1}")
            line = f.readline()
            new_cases, size_cases = line_to_cases(line)
            size_missing.append(size_cases)
            num_cases = new_cases.shape[0]
            new_probs = get_prob(num_cases)
            cases = np.vstack([cases, new_cases])
            #print('case dim: ', cases.shape)
            probs = np.hstack([probs, new_probs])
            #print('prob dim:', probs.shape)

        #print('total case dim: ', cases.shape)
        #print('total prob dim:', probs.shape)
    return cases, probs, size_missing

#cases, probs = get_initial_cases('data/dataset1/train-p-1.txt')
#cases2, probs2 = get_initial_cases('data/dataset1/train-p-2.txt')


def get_cpt_M_step(factor, data, probs):
    """

    :param factor: a CPT
    :param data: the samples
    :param probs: the probs of samples
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
            count_Y = np.sum(probs)
        else:
            for i in range(num_parent):
                rv = rvs[i]
                value = index[i]
                condition_Y = condition_Y & (data[:, rv.name] == value)
            #print("condition Y: ", condition_Y)
            weights = condition_Y * probs
            count_Y = np.sum(weights)

        condition = True
        for i in range(num_rvs):
            rv = rvs[i]
            value = index[i]
            condition = condition & (data[:, rv.name]==value)

        weights_XY = condition * probs
        count_XY = np.sum(weights_XY)
        #print("count XY: ", count_XY)
        #print("count_Y: ", count_Y)
        # use small number to smooth
        CPT[index] = (count_XY + 1e-5) / (count_Y + 2e-5)


# rvs, fs = load('sample_bayes.uai')
# cases, probs = get_initial_cases('sample_p.txt')
# update CPTs
def M_step(fs, cases, probs):
    for i, f in fs.items():
        get_cpt_M_step(f, cases, probs)
        # if i < 5:
        #     print(f.table)

# M_step(fs, cases, probs)
# for i, f in fs.items():
#     print(f.table)

## update likelihood of cases
def E_step(fs, cases, probs, size_missing):
    for i, case in enumerate(cases):
        loglike_case = log_likelihood_per_sample(case, fs)
        likelihood = np.exp(loglike_case)
        probs[i] = likelihood
        #print("case: ", case)
        #print(f"probs{i}", probs[i])

    begin = 0
    for window_size in size_missing:
        end = begin + window_size
        probs[begin:end] /= np.sum(probs[begin:end])
        #print("after normaliztion",np.sum(probs[begin:end]))
        begin = end

    return probs
# print("Before updating: ", probs[5])
# probs = E_step(fs, cases, probs)
# print("After updating: ", probs[5])
#
# print("**********")
# M_step(fs, cases, probs)
# print("updating parameters")
#
# for i, f in fs.items():
#     print(f.table)

def EM(uai_file, train_data):
    rvs, fs = load(uai_file)
    cases, probs, size_missing = get_initial_cases(train_data)
    # print(size_missing)
    # initialize the parameter
    M_step(fs, cases, probs)
    #print("initial probs: ", probs[0:10])
    # for i, f in fs.items():
    #     print(f.table)
    for i in range(20):
        probs = E_step(fs, cases, probs, size_missing)
        M_step(fs, cases, probs)
        #print(f"{i}th probs: ", np.sum(probs[0:32]))
    # print("result: ")
    # for i, f in fs.items():
    #     print(f.table)
    return fs

# true_rvs, true_fs = get_original_model('sample_bayes.uai')
# fs = EM('sample_bayes.uai', 'sample_p.txt')
# dif_sum('sample_test.txt', fs, true_fs)

def run_em_5times(uai_file, train_data, test_data, num_iter=5):
    print("------------------------")
    start_time = time.time()
    true_rvs, true_fs = get_original_model(uai_file)
    diffs = []
    for i in range(num_iter):
        print(f"The {i+1}th time to run EM:")
        fs = EM(uai_file, train_data)
        dif = dif_sum(test_data, fs, true_fs)
        print('dif: ', dif)
        diffs.append(dif)
    avg_dif = np.mean(diffs)
    avg_dif = round(avg_dif, 4)
    std_dif = np.std(diffs)
    std_dif = round(std_dif, 4)
    average_time = (time.time() - start_time)
    average_time = str(round(average_time, 3))
    print(f"The running time of EM is: {average_time} s.")
    print("------------------------")

    return avg_dif, std_dif

# avg_dif, std_dif = run_em_5times('sample_bayes.uai', 'sample_p.txt', 'sample_test.txt')
# avg_dif, std_dif = run_em_5times('data/dataset1/1.uai', "data/dataset1/train-p-1.txt", "data/dataset1/test.txt")
# print("avg dif: ", avg_dif)
# print("std dif:", std_dif)

em_result = "em_result2.txt"

uai_files = ['data/dataset1/1.uai', 'data/dataset2/2.uai', 'data/dataset3/3.uai']
em_train_files = ["train-p-1.txt", "train-p-2.txt", "train-p-3.txt"]
em_train_files2 = ["train-p-4.txt"]

def train_dataset(uai_file, train_dir, test_data, result_file, em_train_files):
    print(f"Training {os.path.basename(uai_file)} :")
    for train_file in em_train_files:
        print(f"Using {train_file}: ")
        train_data = train_dir + "/" + train_file

        avg_dif, std_dif = run_em_5times(uai_file, train_data, test_data)
        with open(result_file, 'a') as f:
            f.write(os.path.basename(uai_file) + "\t" + train_file + "\t")
            f.write("Mean dif = " + str(avg_dif) + "\t" + "Std: "+ str(std_dif) + "\n")

def main():
    train_dataset('data/dataset1/1.uai', 'data/dataset1', 'data/dataset1/test.txt', em_result, em_train_files)
    train_dataset('data/dataset2/2.uai', 'data/dataset2', 'data/dataset2/test.txt', em_result, em_train_files)
    train_dataset('data/dataset3/3.uai', 'data/dataset3', 'data/dataset3/test.txt', em_result, em_train_files)
    train_dataset('data/dataset1/1.uai', 'data/dataset1', 'data/dataset1/test.txt', em_result, em_train_files2)
    train_dataset('data/dataset2/2.uai', 'data/dataset2', 'data/dataset2/test.txt', em_result, em_train_files2)
    train_dataset('data/dataset3/3.uai', 'data/dataset3', 'data/dataset3/test.txt', em_result, em_train_files2)

if __name__ == '__main__':
    main()
    # avg_dif, std_dif = run_em_5times("sample_bayes.uai", 'sample_p.txt', 'sample_test.txt')
    # EM("data/dataset1/1.uai", 'data/dataset1/train-p-1.txt')
    # avg_dif, std_dif = run_em_5times("data/dataset1/1.uai", 'data/dataset1/train-p-1.txt', 'data/dataset1/test.txt')
    # avg_dif, std_dif = run_em_5times("data/dataset1/1.uai", 'data/dataset1/train-p-2.txt', 'data/dataset1/test.txt')
    # EM("sample_bayes.uai", 'sample_p.txt')
    pass
    # with open("em_result.txt", 'a') as f:
    #     f.write(os.path.basename("sample_bayes.uai") + "\t" + 'sample_p.txt' + "\t")
    #     f.write("Mean dif = " + str(avg_dif) + "\t" + "Std: " + str(std_dif) + "\n")