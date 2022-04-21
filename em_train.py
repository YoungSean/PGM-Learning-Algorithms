import numpy as np
from collections import deque
from uai_loader import load

# get n numbers from 0 to 1. They sum to 1.
# used for parameter initialization
def get_prob(n):
    x = np.random.randint(10, size=n)
    # softmax
    return np.exp(x)/sum(np.exp(x))


# probs = get_N_prob(2)
# print(probs)
# print(type(probs))

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

    return np.array(cases)

# line = '0 ? 1 ? ?'
#
# cases = line_to_cases(line)
# print(cases)

def get_initial_cases(train_file):
    with open(train_file, 'r') as f:
        first_line = f.readline()
        num_vr, num_rows = first_line.split()
        num_rows = int(num_rows)
        first_case = f.readline()
        cases = line_to_cases(first_case)
        num_cases = cases.shape[0]
        #print("num_cases: ", num_cases)
        probs = get_prob(num_cases)
        #print(probs)

        for i in range(num_rows - 1):
            line = f.readline()
            new_cases = line_to_cases(line)
            num_cases = new_cases.shape[0]
            new_probs = get_prob(num_cases)
            cases = np.vstack([cases, new_cases])
            #print('case dim: ', cases.shape)
            probs = np.hstack([probs, new_probs])
            #print('prob dim:', probs.shape)

        print('case dim: ', cases.shape)
        print('prob dim:', probs.shape)
    return cases, probs

# cases, probs = get_initial_cases('data/dataset1/train-p-1.txt')
cases, probs = get_initial_cases('sample_p.txt')

def get_cpt_M_step(factor, data, probs):
    """

    :param factor: a CPT
    :param data: the samples
    :param probs: the probs of samples
    :return:
    """
    # factor.table = np.zeros_like(factor.table)
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

            weights = condition_Y * probs
            count_Y = np.sum(weights)

        condition = True
        for i in range(num_rvs):
            rv = rvs[i]
            value = index[i]
            condition = condition & (data[:, rv.name]==value)

        weights_XY = condition * probs
        count_XY = np.sum(weights_XY)
        print("count XY: ", count_XY)
        print("count_Y: ", count_Y)
        # use small number to smooth
        CPT[index] = (count_XY + 1e-5) / (count_Y + 2e-5)

rvs, fs = load('sample_bayes.uai')

for i,f in fs.items():
    get_cpt_M_step(f, cases, probs)

for i, f in fs.items():
    print(f.table)