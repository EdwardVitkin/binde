#       Library with Appearance Differential Expression analysis functions
#
#   Author:     Edward Vitkin
#   email:      edward.vitkin@gmail.com
#   version:    14 Dec, 2023
#
import numpy as np
from scipy import stats

#   Hypergeometric distribution based Test on data statistics:
#   INPUTS:
#       Na - number of experiments from type A
#       cA - number of experiments from type A, where the molecule of interest was observed
#       Nb - number of experiments from type B
#       cB - number of experiments from type B, where the molecule of interest was observed
def hg_binde_test_sizes(Na, cA, Nb, cB):
    population_sz = Na + Nb
    groupA_sz = Na
    groupB_sz = cA + cB
    intrsct_sz = cA
    pvalue = stats.hypergeom.sf(intrsct_sz-1, population_sz, groupA_sz, groupB_sz)

    return pvalue

#   Hypergeometric distribution based Test on experiment binary indicator vectors
#   INPUTS:
#       grpA - binary appearance indicator of molecule in Group A experiments
#       grpB - binary appearance indicator of molecule in Group B experiments
#           values:
#               0 (False) - molecule was NOT observed in the experiment
#               1 (True) - molecule was observed in the experiment
def hg_binde_test_vec(indicators_grpA, indicators_grpB):
    Na, Nb = len(indicators_grpA), len(indicators_grpB)
    cA, cB = np.count_nonzero(indicators_grpA), np.count_nonzero(indicators_grpB)
    return hg_binde_test_sizes(Na, cA, Nb, cB)

#   Hypergeometric distribution based Test on values:
#   INPUTS:
#       values_grpA - values of experiments from type A
#       values_grpB - values of experiments from type A
def hg_binde(values_grpA, values_grpB):
    indicators_grpA = np.array([v>0 for v in values_grpA])
    indicators_grpB = np.array([v>0 for v in values_grpB])
    return hg_binde_test_vec(indicators_grpA, indicators_grpB)


#   Proportion based Test on experiment binary indicator vectors
#   INPUTS:
#       grpA - binary appearance indicator of molecule in Group A experiments
#       grpB - binary appearance indicator of molecule in Group B experiments
#           values:
#               0 (False) - molecule was NOT observed in the experiment
#               1 (True) - molecule was observed in the experiment
#       alternative - tested alternatives:
#           two-sided [DEFAULT] - Checks if proportion of appearances of Group A is DIFFERENT of that of Group B
#           less - Checks if proportion of appearances of Group A is LOWER of that of Group B
#           greater [DEFAULT] - Checks if proportion of appearances of Group A is HIGHER of that of Group B
def proportion_binde_test(indicators_grpA, indicators_grpB, alternative='two-sided'):
    nA = indicators_grpA.shape[0]
    nB = indicators_grpB.shape[0]
    hitA = np.count_nonzero(indicators_grpA, axis=0)
    hitB = np.count_nonzero(indicators_grpB, axis=0)
    pA = hitA / nA
    pB = hitB / nB

    zscore, pvalue = np.zeros(shape=pA.shape), np.ones(shape=pA.shape)

    if np.all((pA+pB)<=0) or np.all((pA+pB)>=2):        #   all-zero OR all-one probabilities
        return zscore, pvalue

    p = (hitA+hitB) / (nA+nB)
    q = 1 - p
    mu__pApB = 0
    sigma_pApB = np.sqrt( p*q/nA + p*q/nB )
    non_zero_sigma = sigma_pApB != 0

    data_mu_pApB = pA - pB

    zscore[non_zero_sigma] = ( data_mu_pApB[non_zero_sigma] - mu__pApB) / sigma_pApB[non_zero_sigma]

    if alternative=='two-sided':
        pvalue[non_zero_sigma] = 2*stats.norm.sf(abs(zscore[non_zero_sigma]))
    elif alternative=='less':
        pvalue[non_zero_sigma] = stats.norm.sf(-zscore[non_zero_sigma])
    elif alternative=='greater':
        pvalue[non_zero_sigma] = stats.norm.sf(zscore[non_zero_sigma])

    return zscore, pvalue

#   Proportion based Test on values:
#   INPUTS:
#       values_grpA - values of experiments from type A
#       values_grpB - values of experiments from type A
def proportion_binde(values_grpA, values_grpB,alternative='two-sided'):
    indicators_grpA = np.array([v>0 for v in values_grpA])
    indicators_grpB = np.array([v>0 for v in values_grpB])

    return proportion_binde_test(indicators_grpA, indicators_grpB,alternative=alternative)


def usage_example1():
    grp_1 = np.zeros(shape=(50))
    grp_1[:32] = 1
    grp_2 = np.zeros(shape=(60))
    grp_2[:18] = 1

    ttl_str = lambda grp_1,grp_2: f"({np.count_nonzero(grp_1)}/{len(grp_1)} vs {np.count_nonzero(grp_2)}/{len(grp_2)} successes)"

    print('Proportion-based test (zscore,pvalue)')
    print(f"\t{ttl_str(grp_1, grp_2)}, alternative=two-sided = {proportion_binde_test(grp_1, grp_2)}")
    print(f"\t{ttl_str(grp_2, grp_1)}, alternative=two-sided = {proportion_binde_test(grp_2, grp_1)}")
    print(f"\t{ttl_str(grp_1, grp_2)}, alternative=less = {proportion_binde_test(grp_1, grp_2, 'less')}")
    print(f"\t{ttl_str(grp_1, grp_2)}, alternative=greater = {proportion_binde_test(grp_1, grp_2, 'greater')}")

    print('HG-based test (pvalue)')
    print(f"\t{ttl_str(grp_1, grp_2)} = {hg_binde_test_vec(grp_1, grp_2)}")
    print(f"\t{ttl_str(grp_2, grp_1)} = {hg_binde_test_vec(grp_2, grp_1)}")
    return
def usage_example2():
    grp_1 = 1000*np.random.rand(50)
    grp_1[32:] = 0
    grp_2 = 1000*np.random.rand(60)
    grp_2[18:] = 0

    ttl_str = lambda grp_1,grp_2: f"({np.count_nonzero(grp_1)}/{len(grp_1)} vs {np.count_nonzero(grp_2)}/{len(grp_2)} successes)"

    print('Proportion-based test (zscore,pvalue)')
    print(f"\t{ttl_str(grp_1, grp_2)}, alternative=two-sided = {proportion_binde(grp_1, grp_2)}")
    print(f"\t{ttl_str(grp_2, grp_1)}, alternative=two-sided = {proportion_binde(grp_2, grp_1)}")
    print(f"\t{ttl_str(grp_1, grp_2)}, alternative=less = {proportion_binde(grp_1, grp_2, 'less')}")
    print(f"\t{ttl_str(grp_1, grp_2)}, alternative=greater = {proportion_binde(grp_1, grp_2, 'greater')}")

    print('HG-based test (pvalue)')
    print(f"\t{ttl_str(grp_1, grp_2)} = {hg_binde(grp_1, grp_2)}")
    print(f"\t{ttl_str(grp_2, grp_1)} = {hg_binde(grp_2, grp_1)}")
    return


if __name__ == '__main__':
    usage_example1()
    usage_example2()
