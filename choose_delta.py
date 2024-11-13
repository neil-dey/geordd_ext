import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(0)

delta = 5
epsilon = 0.1

lambda_0 = 5
tau = 10

deltas = []
tau_estimates = []
for mc_iter in range(10):
    print(mc_iter)
    # Generate some sample data
    pos_points = [st.expon.rvs(scale = 1/(lambda_0 + tau))]
    while pos_points[-1] < delta:
        pos_points.append(pos_points[-1] + st.expon.rvs(scale = 1/(lambda_0+tau)))

    neg_points = [-1*st.expon.rvs(scale = 1/lambda_0)]
    while neg_points[-1] > -1*delta:
        neg_points.append(neg_points[-1] - st.expon.rvs(scale = 1/lambda_0))

    epsilons = np.array([(n+1)*epsilon for n in range(int(delta/epsilon))])
    thetahats = np.array([len([x for x in pos_points if x < (n+1)*epsilon]) - len([x for x in neg_points if x > -(n+1)*epsilon]) for n in range(int(delta/epsilon))])

    pvals = []
    for k in range(2, int(delta/epsilon)):
        # Estimate the distribution of beta hat
        covs = np.array([(i+1)*epsilon for i in range(k)])
        Xt = np.stack((np.ones(k)*1.0, covs))
        hat = np.linalg.inv(Xt @ Xt.T) @ Xt
        theory = []

        pos_mean = len([x for x in pos_points if x < (k+1)*epsilon])/((k+1)*epsilon)
        neg_mean = len([x for x in neg_points if x > -(k+1)*epsilon])/((k+1)*epsilon)

        pval_iters = 100
        for _ in range(pval_iters):
            theory.append(sum([(12*i-6*k+6)/((k-1)*k*(k+1)*epsilon) * (st.poisson.rvs((pos_mean)*(i+1)*epsilon) - st.poisson.rvs(neg_mean*(i+1)*epsilon)) for i in range(k)]))

        # Compute the actual beta hat
        beta1 = np.linalg.lstsq(np.vstack([np.ones(k), epsilons[:k]]).T, thetahats[:k], rcond = None)[0][1]

        if beta1 > 0:
            pval = len([beta for beta in theory if beta > beta1 or beta < -1*beta1])/pval_iters
        else:
            pval = len([beta for beta in theory if beta < beta1 or beta > -1*beta1])/pval_iters
        pvals.append(pval)

    pvals = st.false_discovery_control(pvals)
    k = 2
    for idx, pval in enumerate(pvals):
        if pval < 0.05:
            break
        k += 1

    #print("    ", epsilons[k-1])
    #print("    ", thetahats[k-1]/epsilons[k-1])
    deltas.append(epsilons[k-1])
    tau_estimates.append(thetahats[k-1]/epsilons[k-1])

print(deltas)
print(np.mean(deltas))
print()
print(tau_estimates)
print(np.mean(tau_estimates))
print(np.mean([(t - tau)**2 for t in tau_estimates]))
