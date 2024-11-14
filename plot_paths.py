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

beta1s = []
thetas = []
for mc_iter in range(5):
    pos_points = [st.expon.rvs(scale = 1/(lambda_0 + tau))]
    while pos_points[-1] < delta:
        pos_points.append(pos_points[-1] + st.expon.rvs(scale = 1/(lambda_0+tau + confounding(pos_points[-1]))))

    neg_points = [-1*st.expon.rvs(scale = 1/lambda_0)]
    while neg_points[-1] > -1*delta:
        neg_points.append(neg_points[-1] - st.expon.rvs(scale = 1/lambda_0))

    epsilons = np.array([(n+1)*epsilon for n in range(int(delta/epsilon))])
    thetahats = np.array([len([x for x in pos_points if x < (n+1)*epsilon]) - len([x for x in neg_points if x > -(n+1)*epsilon]) for n in range(int(delta/epsilon))])
    tauhats = [theta/epsilon for (theta, epsilon) in zip(thetahats, epsilons)]

    beta1 = np.linalg.lstsq(np.vstack([np.ones(len(epsilons)), epsilons]).T, tauhats, rcond = None)[0]
    beta1s.append(beta1[1])

    plt.plot(epsilons, tauhats)
    #plt.plot(epsilons, [beta1[0] + beta1[1]*e for e in epsilons])

plt.savefig("tauhat_plot_linear.png")
