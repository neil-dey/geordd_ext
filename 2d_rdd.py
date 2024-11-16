import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d # 3d plots

np.random.seed(0)

x_domain = 30 # range [-x_dom, x_dom]
y_domain = 10 # range [-y_dom, y_dom]

epsilon = 0.1
lambda_0 = 1
tau = 1

def confounder(x):
    if x >= 0:
        c = tau/2*np.exp(-x)-tau/2
        if x < 6:
            c += 2*tau*np.exp(-1/(1-(x/3-1)**2))
        return c
    return -1*confounder(-x)

def generate_data(granularity = epsilon/10):
    pos_points = []
    neg_points = []
    for i in range(int(x_domain/granularity)):
        x = (i+0.5)*granularity
        num_points = st.poisson.rvs((lambda_0 + tau + confounder(x)) * granularity * (2*y_domain))
        pos_points.extend([(st.uniform.rvs(i*granularity, scale = granularity), st.uniform.rvs(-y_domain, 2*y_domain)) for _ in range(num_points)])

        num_points = st.poisson.rvs((lambda_0 + confounder(-x)) * granularity * (2*y_domain))
        neg_points.extend([(-1*st.uniform.rvs(i*granularity, scale = granularity), st.uniform.rvs(-y_domain, 2*y_domain)) for _ in range(num_points)])

    return pos_points, neg_points

# Plot tauhat as a function of delta, if you only use the endpoint
def plot_tauhat_path(num_paths = 3):
    # Plot sample tauhats
    for _ in range(num_paths):
        pos_points, neg_points = generate_data()

        epsilons = np.array([(n+1)*epsilon for n in range(int(x_domain/epsilon))])
        thetahats = np.array([len([x for (x, y) in pos_points if x < (n+1)*epsilon]) - len([x for (x, y) in neg_points if x > -(n+1)*epsilon]) for n in range(int(x_domain/epsilon))])
        tauhats = [theta/(epsilon*2*y_domain) for (theta, epsilon) in zip(thetahats, epsilons)]

        plt.plot(epsilons, tauhats)

    # Plot expected value of tauhat
    plt.plot(epsilons, [tau + confounder(r) - confounder(-r) for r in epsilons], color = 'black', linestyle = 'dashed')
    plt.xlabel(r"$\delta$")
    plt.ylabel(r"$\widehat{\tau}$")
    plt.savefig("tauhat_plot_2D.png")
    plt.clf()

def plot_intensity():
    ax = plt.figure().add_subplot(projection='3d')
    xs = np.array([np.linspace(-x_domain, x_domain, num = 100) for _ in range(100)])
    ys = np.array([np.array([y for _ in range(100)]) for y in np.linspace(-y_domain, y_domain, num = 100)])
    zs = np.zeros((100, 100))

    for i in range(100):
        for j in range(100):
            zs[i][j] = lambda_0 + tau + confounder(xs[i][j]) if xs[i][j] > 0 else lambda_0 + confounder(xs[i][j])

    ax.plot_surface(xs, ys, zs, edgecolor='royalblue', lw=0.5, alpha = 0.3)

    ps, ns = generate_data()
    xs, ys = zip(*ps)
    ax.scatter(xs, ys, zs=0, zdir='z')
    xs, ys = zip(*ns)
    ax.scatter(xs, ys, zs=0, zdir='z')
    ax.set(xlabel = r"$x$", ylabel = r"$y$", zlabel = r"$\widehat{\lambda}(x, y)$")

    plt.show()
    plt.clf()

# Code to show a progress bar from https://stackoverflow.com/a/34482761
import sys
import time
def progressbar(it, prefix="", size=60, out=sys.stdout):
    count = len(it)
    start = time.time()
    def show(j):
        x = int(size*j/count)
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
    show(0.1)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def choose_delta(num_paths = 100):
    deltas = []
    tau_estimates = []
    end_tau_estimates = []
    beg_tau_estimates = []
    avg_tau_estimates = []

    for _ in progressbar(range(num_paths)):
        pos_points, neg_points = generate_data()

        epsilons = np.array([(n+1)*epsilon for n in range(int(x_domain/epsilon))])
        thetahats = np.array([len([x for (x, y) in pos_points if x < (n+1)*epsilon]) - len([x for (x, y) in neg_points if x > -(n+1)*epsilon]) for n in range(int(x_domain/epsilon))])
        tauhats = [theta/(epsilon*2*y_domain) for (theta, epsilon) in zip(thetahats, epsilons)]

        # We compute p-values backwards, from largest delta to smallest
        ks = [k for k in range(2, int(x_domain/epsilon))][::-10]
        idx = None
        for k in ks:
            # Estimate the distribution of beta hat
            theory = []
            pos_mean = len([x for (x, y) in pos_points if x < k*epsilon])/(k*epsilon * 2*y_domain)
            neg_mean = len([x for (x, y) in neg_points if x > -k*epsilon])/(k*epsilon * 2*y_domain)
            pval_iters = 100
            for _ in range(pval_iters):
                theory.append(sum([(12*i-6*k+6)/((k-1)*k*(k+1)*epsilon * (i+1)*epsilon) * (st.poisson.rvs(pos_mean*(i+1) * epsilon) - st.poisson.rvs(neg_mean*(i+1) * epsilon)) for i in range(k)]))

            # Compute p-value for H_0: beta = 0
            beta1 = np.linalg.lstsq(np.vstack([np.ones(k), epsilons[:k]]).T, tauhats[:k], rcond = None)[0][1]
            theory = sorted(theory)
            quantile = np.searchsorted(theory, beta1)/len(theory)
            if quantile > 0.5:
                quantile = 1 - quantile
            pval = 2*quantile

            # Stop once we fail to reject H_0
            if pval > 0.05:
                idx = k
                break

        k = idx
        deltas.append(epsilons[k])
        tau_estimates.append(np.mean([thetahats[0:k+1]/(epsilons[0:k+1]*2*y_domain)]))
        beg_tau_estimates.append(thetahats[0]/(epsilons[0]*2*y_domain))
        end_tau_estimates.append(thetahats[-1]/(epsilons[-1]*2*y_domain))
        avg_tau_estimates.append(np.mean(thetahats/(epsilons*2*y_domain)))

    print(deltas)
    print("Final results")
    print("Avg. Delta:", np.mean(deltas))
    print()
    print("Proposal tau Bias:", np.mean(tau_estimates) - tau)
    print("Proposal tau MSE:", np.mean([(t - tau)**2 for t in tau_estimates]))
    print()
    print("Beginning tau Bias:", np.mean(beg_tau_estimates) - tau)
    print("Beginning tau MSE", np.mean([(t - tau)**2 for t in beg_tau_estimates]))
    print()
    print("End tau Bias:", np.mean(end_tau_estimates) - tau)
    print("End tau MSE", np.mean([(t - tau)**2 for t in end_tau_estimates]))
    print()
    print("Average tau Bias:", np.mean(avg_tau_estimates) - tau)
    print("Average tau MSE", np.mean([(t - tau)**2 for t in avg_tau_estimates]))

plot_intensity()
plot_tauhat_path()
choose_delta()
