import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True # Use LaTeX in plot labels

np.random.seed(1)

# Constants
delta = 20
epsilon = 0.2
lambda_0 = 5
tau = 10

# The different confounding functions as a function of location
def no_confounder(x):
    return 0
def relu_confounder(x):
    if x > 0:
        return x
    return 0
def tanh_confounder(x):
    if x >= 0:
        return tau/(2*(1+np.tanh(tau)))*(np.tanh(-tau) - np.tanh(x - tau))
    return -tau/(2*(1+np.tanh(tau)))*(np.tanh(-tau) - np.tanh(-x - tau))

# Generates sample data given a confounder
def generate_data(confounder):
        pos_points = [st.expon.rvs(scale = 1/(lambda_0 + tau))]
        while pos_points[-1] < delta:
            pos_points.append(pos_points[-1] + st.expon.rvs(scale = 1/(lambda_0 + tau + confounder(pos_points[-1]))))

        neg_points = [-1*st.expon.rvs(scale = 1/lambda_0)]
        while neg_points[-1] > -1*delta:
            neg_points.append(neg_points[-1] - st.expon.rvs(scale = 1/(lambda_0 + confounder(neg_points[-1]))))

        return pos_points, neg_points


# Plot tauhat as a function of delta, if you only use the endpoint
def plot_tauhat_path(confounder, num_paths = 3):
    # Plot sample tauhats
    for _ in range(num_paths):
        pos_points, neg_points = generate_data(confounder)

        epsilons = np.array([(n+1)*epsilon for n in range(int(delta/epsilon))])
        thetahats = np.array([len([x for x in pos_points if x < (n+1)*epsilon]) - len([x for x in neg_points if x > -(n+1)*epsilon]) for n in range(int(delta/epsilon))])
        tauhats = [theta/epsilon for (theta, epsilon) in zip(thetahats, epsilons)]

        plt.plot(epsilons, tauhats)

    # Plot expected value of tauhat
    plt.plot(epsilons, [tau + confounder(r) - confounder(-r) for r in epsilons], color = 'black', linestyle = 'dashed')
    plt.xlabel(r"$\delta$")
    plt.ylabel(r"$\widehat{\tau}$")
    plt.savefig("tauhat_plot_" + confounder.__name__ + ".png")
    plt.clf()

# Plot the intensity function
def plot_intensity(confounder, num_paths = 3):
    # Set up the x-axis
    granularity = 100
    xs = np.linspace(-delta, delta, num = granularity)
    # We want a discontinuity at the origin---force 0 to be in the x-axis space
    if xs[granularity//2] != 0:
        xs = np.insert(xs, granularity//2, 0)

    # Plot sample intensities
    for _ in range(num_paths):
        pos_points, neg_points = generate_data(confounder)
        plt.plot(xs, [len([p for p in neg_points if p > x])/-x if x < 0 else len([p for p in pos_points if p < x])/x if x > 0 else np.nan for x in xs])

    # Plot true intensity function
    plt.plot(xs, [lambda_0 + confounder(x) if x < 0 else lambda_0 + tau + confounder(x) if x > 0 else np.nan for x in xs], color = 'black', linestyle = 'dashed')
    plt.xlabel("$x$")
    plt.ylabel(r"$\widehat{\lambda}(x)$")
    plt.savefig("intensity_plot_" + confounder.__name__ + ".png")
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

# Compute the bias/MSE of various estimation procedures
def choose_delta(confounder, num_paths = 100):
    deltas = []
    tau_estimates = []
    end_tau_estimates = []
    beg_tau_estimates = []
    avg_tau_estimates = []

    for _ in progressbar(range(num_paths)):
        pos_points, neg_points = generate_data(confounder)

        epsilons = np.array([(n+1)*epsilon for n in range(int(delta/epsilon))])
        thetahats = np.array([len([x for x in pos_points if x < (n+1)*epsilon]) - len([x for x in neg_points if x > -(n+1)*epsilon]) for n in range(int(delta/epsilon))])
        tauhats = [thetahat/epsilon for (thetahat, epsilon) in zip(thetahats, epsilons)]

        # We compute p-values backwards, from largest delta to smallest
        ks = [k for k in range(2, int(delta/epsilon))][::-1]
        idx = None
        for k in ks:
            # Estimate the distribution of beta hat
            theory = []
            pos_mean = len([x for x in pos_points if x < k*epsilon])/(k*epsilon)
            neg_mean = len([x for x in neg_points if x > -k*epsilon])/(k*epsilon)
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
        tau_estimates.append(np.mean([thetahats[0:k+1]/epsilons[0:k+1]]))
        beg_tau_estimates.append(thetahats[0]/epsilons[0])
        end_tau_estimates.append(thetahats[-1]/epsilons[-1])
        avg_tau_estimates.append(np.mean(thetahats/epsilons))

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

"""
plot_tauhat_path(no_confounder)
plot_tauhat_path(relu_confounder)
plot_tauhat_path(tanh_confounder)

plot_intensity(no_confounder)
plot_intensity(relu_confounder)
plot_intensity(tanh_confounder)
"""

#choose_delta(no_confounder)
#choose_delta(relu_confounder)
choose_delta(tanh_confounder)
