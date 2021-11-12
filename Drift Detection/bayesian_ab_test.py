import numpy as np
import pandas as pd
import pymc3 as pmc
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from collections.abc import Iterable


def train_bow_model(text_input):
    """
    Fits a bag of words (BoW) model to the text

    :param text_input: list of documents

    :return: vectorizer, np array of vectorized text
    """
    assert isinstance(text_input, Iterable) and not isinstance(text_input, str)
    vectorizer = CountVectorizer(stop_words="english")
    vectorizer.fit(text_input)
    return vectorizer, vectorizer.transform(text_input)


def batch(iterable, batch_size=1):
    """
    Used to create batches of equal size, batch_size

    Example usage:
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # list of data

        for x in batch(data, 3):
            print(x)

        # Output

        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9, 10]
    """
    iterable_len = len(iterable)
    for ndx in range(0, iterable_len, batch_size):
        yield iterable[ndx:min(ndx + batch_size, iterable_len)]


def get_mean_and_std(train, test):
    """
    Calculates the mean & standard deviation for the train and test datasets

    :param train: numpy array of training data
    :param test: numpy array of test data
    """
    train_mu = np.mean(train, axis=0)
    train_sigma = np.std(train, axis=0)
    test_mu = np.mean(test, axis=0)
    test_sigma = np.std(test, axis=0)
    return train_mu, train_sigma, test_mu, test_sigma


twenty_train_full = fetch_20newsgroups(
    subset='train',
    shuffle=True,
    random_state=14
)
twenty_test_full = fetch_20newsgroups(
    subset='test',
    shuffle=True,
    random_state=14
)

# implement streaming-like functionality by batching
batch_size = 5000
nbr_batches = int(np.ceil(len(twenty_test_full.data) / batch_size))
current_batch = 1
for batch_indices in batch(iterable=range(len(twenty_test_full.data)), batch_size=batch_size):
    progress = round(100 * current_batch / nbr_batches, 2)
    print(f"Streaming Simulation progress: {progress}%")

    # create the train & test sets for this batch
    twenty_test = pd.DataFrame({
        "data": [twenty_test_full.data[idx] for idx in batch_indices]
    })
    if current_batch > 1:
        twenty_train = pd.DataFrame({
            "data": twenty_train_full.data + twenty_test_full.data[:max(batch_indices) + 1]
        })
    else:
        twenty_train = pd.DataFrame({"data": twenty_train_full.data})

    # train a bow model - this will not be needed in prod, bc will have already been trained
    bow_model, twenty_train_bow = train_bow_model(text_input=twenty_train.data)

    # vectorize train and test data - this will not be needed in prod, bc will receive transformed input
    twenty_train_vect = bow_model.transform(twenty_train.data)
    twenty_test_vect = bow_model.transform(twenty_test.data)

    train_mu, train_sigma, test_mu, test_sigma = get_mean_and_std(
        train=twenty_train_vect.toarray(),
        test=twenty_test_vect.toarray()
    )

    min_tolerance_threshold = 0.25  # prob that effect size has shrunk can't be > 75%
    max_tolerance_threshold = 0.75  # prob that effect size has increased can't be > 75%

    # start of hypothesis test - this will need to be done for every feature and the model output
    with pmc.Model() as model:

        # prior for means of 1 feature (index 0)
        train_mean = pmc.Normal(name='train_mean', mu=train_mu[0], sigma=train_sigma[0])
        test_mean = pmc.Normal(name='test_mean', mu=test_mu[0], sigma=test_sigma[0])

        # prior for nu (degrees of freedom in Student T's PDF) with lambda = 30 to balance
        #   nearly normal with long tailed distributions, and shifted by 1 (because lambda - 1 DoF)
        v = pmc.Exponential(name='v_minus_one', lam=1 / 29.) + 1

        # prior for standard deviations
        sigma_low = 1
        sigma_high = 20
        train_std = pmc.Uniform(name='train_std', lower=sigma_low, upper=sigma_high)
        test_std = pmc.Uniform(name='test_std', lower=sigma_low, upper=sigma_high)

        # transform prior standard deviations to precisions (precision = reciprocal of variance)
        # this will allow specifying lambda in pmc.StudentT instead of sigma, and the spread will converge
        #   towards precision as nu increases
        train_lambda = train_std**-2
        test_lambda = test_std**-2
        train = pmc.StudentT(name='train', nu=v, mu=train_mean, lam=train_lambda, observed=twenty_train_vect.toarray()[:, 0])
        test = pmc.StudentT(name='test', nu=v, mu=test_mean, lam=test_lambda, observed=twenty_test_vect.toarray()[:, 0])

        # calculate effect size (the diff in the means / pooled estimate of the std dev)
        diff_of_means = pmc.Deterministic(name='difference_of_means', var=test_mean - train_mean)
        diff_of_stds = pmc.Deterministic(name='difference_of_stds', var=test_std - train_std)
        effect_size = pmc.Deterministic(name='effect_size', var=diff_of_means / np.sqrt((test_std**2 + train_std**2) / 2))

        # MCMC estimation
        trace = pmc.sample(2000, random_seed=14)

        # determine if retraining needed by scaling the effect size mean between the 94% HDI
        # this will not precisely match the probability values on the plot, but it will be close
        values = pmc.summary(trace[1000:]).loc["effect_size"][["mean", "hdi_3%", "hdi_97%"]].tolist()
        scaler = MinMaxScaler()
        effect_size_mean_scaled_btwn_94hdi = scaler.fit_transform(np.array(values).reshape(-1, 1))[0][0]
        if (
                effect_size_mean_scaled_btwn_94hdi > max_tolerance_threshold
                or effect_size_mean_scaled_btwn_94hdi < min_tolerance_threshold
        ):
            print("Mean has shifted, need re-training.")

        # plot posterior distributions of all parameters
        pmc.plot_posterior(
            trace[1000:],
            var_names=["difference_of_means", "difference_of_stds", "effect_size"],
            ref_val=0,
            color="#87ceeb"
        )
        plt.show()

# how to combine this with evidently, prometheus + grafana
