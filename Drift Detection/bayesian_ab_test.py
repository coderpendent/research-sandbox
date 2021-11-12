import numpy as np
import pandas as pd
import pymc3 as pmc
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine
from collections.abc import Iterable


d1 = "Obama speaks to the media in Illinois"
d2 = "The President addresses the press in Chicago"
#d2 = "Obama speaks to the media in Illinois"


def train_bow_model(text_input):
    assert isinstance(text_input, Iterable) and not isinstance(text_input, str)
    vectorizer = CountVectorizer(stop_words="english")
    vectorizer.fit(text_input)
    return vectorizer, vectorizer.transform(text_input)

# train a bow model
bow_model, d1_bow = train_bow_model(text_input=[d1])
bow_model_vocab = bow_model.get_feature_names()

# vectorize train and test data
vect_trans = bow_model.transform([d1, d2])

# compare train and test data
w_score = wasserstein_distance(
    u_values=vect_trans[0, :].toarray()[0],
    v_values=vect_trans[1, :].toarray()[0]
)
c_score = 1 - cosine(
    u=vect_trans[0, :].toarray(),
    v=vect_trans[1, :].toarray()
)
c_score = 0.0 if np.isnan(c_score) else c_score
# final_score = 1.0 if (w_score + c_score) == 0 else 2 / (w_score + c_score)
final_score = c_score
print("BOW Distribution Similarity:", final_score)

# compare vocab size
test_set_vocab = CountVectorizer(stop_words="english").fit([d2]).get_feature_names()
features_in_common = set(bow_model_vocab).intersection(set(test_set_vocab))
all_features = set(bow_model_vocab).union(set(test_set_vocab))
print(
    "Similarity between train vocab and test vocab:",
    100 - (100 * (len(all_features) - len(features_in_common)) / len(all_features)),
    "%"
)


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
batch_size = 1000
nbr_batches = int(np.ceil(len(twenty_test_full.data) / batch_size))
current_batch = 1
for batch_indices in batch(iterable=range(len(twenty_test_full.data)), batch_size=batch_size):
    progress = round(100 * current_batch / nbr_batches, 2)
    if progress % 10 == 0:
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

    # train a bow model
    bow_model, twenty_train_bow = train_bow_model(text_input=twenty_train.data)
    bow_model_vocab = bow_model.get_feature_names()

    # vectorize train and test data
    twenty_train_vect = bow_model.transform(twenty_train.data)
    twenty_test_vect = bow_model.transform(twenty_test.data)

    def get_mean_and_std(train, test):
        train_mu = np.mean(train, axis=0)
        train_sigma = np.std(train, axis=0)
        test_mu = np.mean(test, axis=0)
        test_sigma = np.std(test, axis=0)
        return train_mu, train_sigma, test_mu, test_sigma

    train_mu, train_sigma, test_mu, test_sigma = get_mean_and_std(
        train=twenty_train_vect.toarray(),
        test=twenty_test_vect.toarray()
    )


    """
    The exponential distribution can take on any value >= 0, and is continuous.  
    The poisson distribution can take on any value >= 0, and is discrete.
    For this data (count data), the poisson is more appropriate, but the exponential 
    distribution could also be used.  The line has been commented out in favor of 
    Poisson.  
    """

    min_tolerance_threshold = 0.25  # prob that effect size has shrunk can't be > 75%
    max_tolerance_threshold = 0.75  # prob that effect size has increased can't be > 75%

    with pmc.Model() as model:
        # prior for means
        train_mean = pmc.Normal(name='train_mean', mu=train_mu[0], sigma=train_sigma[0])
        test_mean = pmc.Normal(name='test_mean', mu=test_mu[0], sigma=test_sigma[0])
        # prior for nu (degrees of freedom in Student T's PDF) with lambda = 30 to balance
        #   nearly normal with long tailed distributions, and shifted by 1 (because lambda - 1 DoF)
        v = pmc.Exponential(name='v_minus_one', lam=1 / 29.) + 1
        # v = pmc.Poisson(name='v_minus_one', mu=1) + 1  # could be used instead of exponential, if testing discrete dist instead of Student T
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
        # the first set of plots is just informational, not helpful in determining differences
        # pmc.plot_posterior(
        #     trace[1000:],
        #     varnames=['train_mean', 'test_mean', 'train_std', 'test_std', 'v_minus_one'],
        #     color='#87ceeb'
        # )
        pmc.plot_posterior(
            trace[1000:],
            var_names=["difference_of_means", "difference_of_stds", "effect_size"],
            ref_val=0,
            color="#87ceeb"
        )
        plt.show()

    # plots are the relative diffs between distributions of train/test, so 0 means no diff
    # effect size dist shows that 65% of the posterior prob is < 0 and 35% of the prob is > 0, meaning
    #   there is a 65% chance the test mean > train mean and a 35% chance the test mean is < train mean


# how to combine this with evidently, prometheus + grafana



