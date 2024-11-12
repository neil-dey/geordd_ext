import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(0)

delta = 50
epsilon = 0.1

lambda_0 = 5
tau = 10

k = 5

beta1s = []
thetas = []
for _ in range(1000):
    pos_points = [st.expon.rvs(scale = 1/(lambda_0 + tau))]
    while pos_points[-1] < delta:
        pos_points.append(pos_points[-1] + st.expon.rvs(scale = 1/(lambda_0+tau)))

    neg_points = [-1*st.expon.rvs(scale = 1/lambda_0)]
    while neg_points[-1] > -1*delta:
        neg_points.append(neg_points[-1] - st.expon.rvs(scale = 1/lambda_0))

    epsilons = np.array([(n+1)*epsilon for n in range(int(delta/epsilon))])
    thetahats = np.array([len([x for x in pos_points if x < (n+1)*epsilon]) - len([x for x in neg_points if x > -(n+1)*epsilon]) for n in range(int(delta/epsilon))])

    beta1 = np.linalg.lstsq(np.vstack([np.ones(k), epsilons[:k]]).T, thetahats[:k], rcond = None)[0]
    beta1s.append(beta1[1])

    k = 500
    thetahats = [theta/epsilon for (theta, epsilon) in zip(thetahats, epsilons)]
    beta1 = np.linalg.lstsq(np.vstack([np.ones(k), epsilons[:k]]).T, thetahats[:k], rcond = None)[0]

    epsilons = epsilons[0:]
    thetahats = thetahats[0:]
    plt.plot(epsilons, thetahats)
    plt.plot(epsilons, [beta1[0] + beta1[1]*e for e in epsilons])
    plt.show()
    exit()


plt.hist(beta1s)


covs = np.array([(i+1)*epsilon for i in range(k)])
Xt = np.stack((np.ones(k)*1.0, covs))
hat = np.linalg.inv(Xt @ Xt.T) @ Xt
print(hat)
theory = []
for _ in range(1000):
    b = 2/(k*(k-1)) * sum([(-3*(k+1)+6*(i+1))/((k+1)*epsilon) * (st.poisson.rvs((lambda_0+tau)*(i+1)*epsilon) - st.poisson.rvs(lambda_0*(i+1)*epsilon)) for i in range(k)])
    theory.append(b)

plt.hist(theory, alpha = 0.5)
plt.show()

print(st.kstest(beta1s, theory))
