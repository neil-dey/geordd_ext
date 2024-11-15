import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(0)

delta = 20
epsilon = 0.2

lambda_0 = 5
tau = 10

def confounding(x):
    return x
    return -tau/2 - tau/2*np.tanh(x-5)
    return 0

deltas = []
tau_estimates = []
bad_tau_estimates = []

for mc_iter in range(10):
    print(mc_iter)
    # Generate some sample data
    pos_points = [st.expon.rvs(scale = 1/(lambda_0 + tau))]
    while pos_points[-1] < delta:
        pos_points.append(pos_points[-1] + st.expon.rvs(scale = 1/(lambda_0+tau + confounding(pos_points[-1]))))

    neg_points = [-1*st.expon.rvs(scale = 1/lambda_0)]
    while neg_points[-1] > -1*delta:
        neg_points.append(neg_points[-1] - st.expon.rvs(scale = 1/lambda_0))

    epsilons = np.array([(n+1)*epsilon for n in range(int(delta/epsilon))])
    thetahats = np.array([len([x for x in pos_points if x < (n+1)*epsilon]) - len([x for x in neg_points if x > -(n+1)*epsilon]) for n in range(int(delta/epsilon))])
    tauhats = [thetahat/epsilon for (thetahat, epsilon) in zip(thetahats, epsilons)]

    pvals = []
    for k in range(2, int(delta/epsilon)):
        # Estimate the distribution of beta hat
        theory = []
        pos_mean = len([x for x in pos_points if x < k*epsilon])/(k*epsilon)
        neg_mean = len([x for x in neg_points if x > -k*epsilon])/(k*epsilon)
        pval_iters = 100
        for _ in range(pval_iters):
            theory.append(sum([(12*i-6*k+6)/((k-1)*k*(k+1)*epsilon * (i+1)*epsilon) * (st.poisson.rvs(pos_mean*(i+1) * epsilon) - st.poisson.rvs(neg_mean*(i+1) * epsilon)) for i in range(k)]))
            #theory.append(sum([(12*i-6*k+6)/((k-1)*k*(k+1)*epsilon * (i+1)*epsilon) * (st.poisson.rvs((lambda_0+tau)*(i+1)*epsilon) - st.poisson.rvs(lambda_0*(i+1)*epsilon)) for i in range(k)]))

        # Compute p-value for H_0: beta = 0
        beta1 = np.linalg.lstsq(np.vstack([np.ones(k), epsilons[:k]]).T, tauhats[:k], rcond = None)[0][1]
        theory = sorted(theory)
        quantile = np.searchsorted(theory, beta1)/len(theory)
        if quantile > 0.5:
            quantile = 1 - quantile
        pval = 2*quantile

        pvals.append(pval)

    # We should investigate what p-value combiner is most appropriate
    pvals = st.false_discovery_control(pvals, method = 'by')
    k = 2
    for idx, pval in enumerate(pvals):
        if pval < 0.05:
            break
        k += 1

    deltas.append(epsilons[k-1])
    tau_estimates.append(thetahats[k-1]/epsilons[k-1])
    bad_tau_estimates.append(thetahats[-1]/epsilons[-1])

print("Final results")
print("Avg. Delta:", np.mean(deltas))
print()
print("Proposal tau Bias:", tau - np.mean(tau_estimates))
print("Proposal tau MSE:", np.mean([(t - tau)**2 for t in tau_estimates]))
print()
print("Naive tau Bias:", tau - np.mean(bad_tau_estimates))
print("Naive tau MSE", np.mean([(t - tau)**2 for t in bad_tau_estimates]))
