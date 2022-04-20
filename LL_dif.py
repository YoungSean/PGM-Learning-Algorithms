import numpy as np

def get_probs_from_CPTs(sample, fs):
    probs = []

    for i, f in fs.items():
        cpt = f.table
        rvs = f.nb
        values = []
        # get the index (variable assigment) of CPT
        for rv in rvs:
            value = sample[rv.name]
            values.append(value)

        values = tuple(values)
        prob = cpt[values]
        # avoid the probability of 0 or small value
        if prob == 0 or prob < 1e-6:
            prob = 1e-6

        probs.append(prob)

    #print("probs for this assignment: ", probs)
    return probs


def log_likelihood(probs):
    result = 0
    for p in probs:
        result += np.log(p)
    return result


# probs = get_probs_from_CPTs([1,1,1], fs)
# result = log_likelihood(probs)
# print("log likelihood: ", result)

def log_likelihood_per_sample(sample, fs):
    probs = get_probs_from_CPTs(sample, fs)
    result = log_likelihood(probs)
    return result
#
# result = log_likelihood_per_sample([1,1,1], fs)
# print("log likelihood from learned model: ", result)
#
# result = log_likelihood_per_sample([1,1,1], true_fs)
# print("log likelihood from true model: ", result)

def diff_per_sample(sample, fs, true_fs):
    LL_learned = log_likelihood_per_sample(sample, fs)
    LL_true = log_likelihood_per_sample(sample, true_fs)
    return np.abs(LL_learned - LL_true)
# print("difference: ", diff_per_sample([1,1,1], fs, true_fs))

def dif_sum(test_file, fs, true_fs):
    # samples = np.loadtxt("data/dataset1/test.txt", skiprows=1, dtype='int')
    samples = np.loadtxt(test_file, skiprows=1, dtype='int')
    total = 0
    for i in range(samples.shape[0]):
        dif_sample = diff_per_sample(samples[i], fs, true_fs)
        print(f"{i}th dif: {dif_sample}")
        total += dif_sample

    print("log likelihood difference = : ", total)
    return total