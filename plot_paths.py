import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(1)

# Constants
delta = 20
epsilon = 0.2
lambda_0 = 5
tau = 10

def no_confounder(x):
    return 0
def relu_confounder(x):
    if x > 0:
        return x
    return 0
def tanh_confounder(x):
    if x >= 0:
        return tau/4*(np.tanh(-5) - np.tanh(x-5))
    return -tau/4*(np.tanh(-5) - np.tanh(-x-5))

def plot_tauhat_path(confounder, num_paths = 3):
    thetas = []
    for _ in range(num_paths):
        pos_points = [st.expon.rvs(scale = 1/(lambda_0 + tau))]
        while pos_points[-1] < delta:
            pos_points.append(pos_points[-1] + st.expon.rvs(scale = 1/(lambda_0 + tau + confounder(pos_points[-1]))))

        neg_points = [-1*st.expon.rvs(scale = 1/lambda_0)]
        while neg_points[-1] > -1*delta:
            neg_points.append(neg_points[-1] - st.expon.rvs(scale = 1/(lambda_0 + confounder(neg_points[-1]))))

        epsilons = np.array([(n+1)*epsilon for n in range(int(delta/epsilon))])
        thetahats = np.array([len([x for x in pos_points if x < (n+1)*epsilon]) - len([x for x in neg_points if x > -(n+1)*epsilon]) for n in range(int(delta/epsilon))])
        tauhats = [theta/epsilon for (theta, epsilon) in zip(thetahats, epsilons)]


        plt.plot(epsilons, tauhats)
        #beta1 = np.linalg.lstsq(np.vstack([np.ones(len(epsilons)), epsilons]).T, tauhats, rcond = None)[0]
        #plt.plot(epsilons, [beta1[0] + beta1[1]*e for e in epsilons])

    plt.plot(epsilons, [tau + confounder(r) for r in epsilons], color = 'black', linestyle = 'dashed')
    plt.savefig("tauhat_plot_" + confounder.__name__ + ".png")
    plt.clf()

plot_tauhat_path(no_confounder)
plot_tauhat_path(relu_confounder)
plot_tauhat_path(tanh_confounder)
