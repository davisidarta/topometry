import numpy as np

def find_var_features(X, plot=True):
    import statsmodels.api as sm
    features_mean = X.mean(1)
    features_sd = X.std(1)

    # Select overdispersed features
    Sorted_indices = np.argsort(features_mean)

    Fit = sm.nonparametric.lowess(features_sd[Sorted_indices], features_mean[Sorted_indices], is_sorted=True, frac = 1./100)

    var_features = features_sd[Sorted_indices] > Fit[:,1]
    print('Selected '+ str(sum(var_features)) + ' variable features.')

    if plot:
        from matplotlib import pyplot as plt

        # Plotting
        plt.scatter(features_mean, features_sd, color = "blue")
        plt.plot(Fit[: ,0], Fit[: ,1], 'red')
        plt.xlabel("Mean")
        plt.ylabel("Standard deviation")
        plt.show()

        plt.scatter(features_mean[var_features], features_sd[var_features], color = "violet")
        plt.scatter(features_mean[var_features == False], features_sd[var_features == False], color = "blue")
        plt.plot(Fit[:,0], Fit[:,1], 'red')
        plt.show()