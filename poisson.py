import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

#np.random.seed(0)

delta = 10
epsilon = 0.1

lambda_0 = 5
tau = 10

k = 5

def confounding(x):
    return 0
    return np.exp(x)

beta1s = []
thetas = []
for mc_iter in range(1000):
    pos_points = [st.expon.rvs(scale = 1/(lambda_0 + tau))]
    while pos_points[-1] < delta:
        pos_points.append(pos_points[-1] + st.expon.rvs(scale = 1/(lambda_0+tau + confounding(pos_points[-1]))))

    neg_points = [-1*st.expon.rvs(scale = 1/lambda_0)]
    while neg_points[-1] > -1*delta:
        neg_points.append(neg_points[-1] - st.expon.rvs(scale = 1/lambda_0))

    epsilons = np.array([(n+1)*epsilon for n in range(int(delta/epsilon))])
    thetahats = np.array([len([x for x in pos_points if x < (n+1)*epsilon]) - len([x for x in neg_points if x > -(n+1)*epsilon]) for n in range(int(delta/epsilon))])
    tauhats = [theta/epsilon for (theta, epsilon) in zip(thetahats, epsilons)]

    #beta1 = np.linalg.lstsq(np.vstack([np.ones(k), epsilons[:k]]).T, thetahats[:k], rcond = None)[0]
    beta1 = np.linalg.lstsq(np.vstack([np.ones(k), epsilons[:k]]).T, tauhats[:k], rcond = None)[0]
    beta1s.append(beta1[1])

    """
    if mc_iter == 1:
        k = int(delta/epsilon)
        #thetahats = [theta/epsilon for (theta, epsilon) in zip(thetahats, epsilons)]
        beta1 = np.linalg.lstsq(np.vstack([np.ones(k), epsilons[:k]]).T, thetahats[:k], rcond = None)[0]
        epsilons = epsilons
        thetahats = thetahats
        plt.plot(epsilons, thetahats)
        plt.plot(epsilons, [beta1[0] + beta1[1]*e for e in epsilons])
        plt.savefig("thetahat_plot.png")
        plt.show()
        exit()
    """


plt.hist(beta1s)


covs = np.array([(i+1)*epsilon for i in range(k)])
Xt = np.stack((np.ones(k)*1.0, covs))
hat = np.linalg.inv(Xt @ Xt.T) @ Xt
theory = []
for _ in range(1000):
    #b = sum([(12*i-6*k+6)/((k-1)*k*(k+1)*epsilon) * (st.poisson.rvs((lambda_0+tau)*(i+1)*epsilon) - st.poisson.rvs(lambda_0*(i+1)*epsilon)) for i in range(k)])
    b = sum([(12*i-6*k+6)/((k-1)*k*(k+1)*epsilon * (i+1)*epsilon) * (st.poisson.rvs((lambda_0+tau)*(i+1)*epsilon) - st.poisson.rvs(lambda_0*(i+1)*epsilon)) for i in range(k)])
    theory.append(b)

plt.hist(theory, alpha = 0.5)
plt.show()

print(st.kstest(beta1s, theory))
