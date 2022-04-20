import numpy as np
from Graph import Graph, F
from uai_loader import load

data = np.loadtxt("data/dataset1/train-f-1.txt", skiprows=1, dtype='int')
#data = np.loadtxt("sample_dataset.txt", skiprows=1, dtype='int')
# print(data)
print(data.shape)
print(data[0])

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

        if count_Y == 0:
            CPT[index] = 0
            continue

        condition = True
        for i in range(num_rvs):
            rv = rvs[i]
            value = index[i]
            condition = condition & (data[:, rv.name]==value)

        count_XY = np.sum(condition)
        #print("count XY: ", count_XY)
        #print("count_Y: ", count_Y)
        CPT[index] = count_XY / count_Y

    print(CPT)

print("-------------")

# rvs, fs = load('sample_bayes.uai')
rvs, fs = load('data/dataset1/1.uai')
print(fs)

# for i,f in fs.items():
#     get_cpt(f, data)
for i in range(5):
    get_cpt(fs[i], data)

