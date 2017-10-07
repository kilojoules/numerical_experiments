import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
np.random.seed = 1

'''
In this problem I have some high-frequency data that I can't
represent perfectly with my model. Specifically, the "observed"
data is generated as a sum of sin waves with specified amplitudes
of frequencies and the model is sum of sin waves with less frequencies
to specify.
'''

# Each sample is 100 time steps
SAMPLE_TIME = 100

# Number of frequencies in the "observed" data
NUM_TRUTH_FREUQENCIES = 5
TRUE_FF_COEFS = np.random.uniform(0, 1, NUM_TRUTH_FREUQENCIES)

# Create "observed" "truth" Data
samples = 100
TRUE_SIGNALS = []
for _ in range(samples):
    start = np.random.uniform(0, 1e3)
    sample = []
    for t in np.linspace(start, start + 1e4, SAMPLE_TIME):
        sample.append(np.sin(TRUE_FF_COEFS*t).sum())
    TRUE_SIGNALS.append([start, sample])

# Define the low-fidelity model to tune.
NUM_LOWFI_FREQ = 3
def lowfi(LOWFI_FREQ, start):
    signal = []
    for t in np.linspace(start, start + 1e4, SAMPLE_TIME):
        signal.append(np.sin(LOWFI_FREQ * t).sum())
    return np.array(signal)

'''
"vanilla" tuning: minimize the sum of the square of the errors between the models
I could now use these to develop some probabalistic discrepency function between the two models.

I use this to create my prior distributions.
'''
def vanilla_fitness(lowfi_freqs):
    err =  0
    for signal in TRUE_SIGNALS:
        err += np.mean(np.array(lowfi(lowfi_freqs, signal[0])) ** 2.)
    return err
from scipy.optimize import minimize as mini
x = mini(vanilla_fitness, np.ones(NUM_LOWFI_FREQ), method='COBYLA')
vanilla_x = x.x
plt.plot(lowfi(x.x, TRUE_SIGNALS[0][0]), label='Model')
plt.plot(TRUE_SIGNALS[0][1], label='Truth')
plt.legend()
plt.savefig('Vanilla_Tuning.pdf')
plt.clf()

'''
Baysian Calibration

I define a fitness function for single samples of the signal

For every unkown parameter in the low-fidelity model, I create a prior using
the known bounds.
'''

def single_fitness(lowfi_freqs, signal_num):
    #signal_num = args['signal_num']
    true_signal = TRUE_SIGNALS[signal_num]
    return np.mean(np.array(lowfi(lowfi_freqs, true_signal[0]) - true_signal[1]))**2

# Create Gaussian priors centered around the some of squares solution
num_probs = 100
x_prior = []
for ii in range(NUM_LOWFI_FREQ):
    tmp_probs = []
    for val in np.linspace(-5, 5, num_probs): # arbitrary bounds
        tmp_probs.append(norm.pdf(val, vanilla_x[ii], 5)) # arbitrary sigma
    tmp_probs = np.array(tmp_probs) / np.sum(tmp_probs)
    x_prior.append(tmp_probs)

x_probs = np.array(x_prior)
for sample in range(samples):
    # p(X) = P(D|X) * P(X)
            x = mini(single_fitness, np.ones(NUM_LOWFI_FREQ), args=(sample,), method='COBYLA')
            x_probs *= np.sum(x_probs[x_probs < x]) / np.sum(x_probs)

#plt.plot(TRUE_SIGNALS[0][1], label='Truth')
#x = np.linspace(-5, 5, num_probs)[max(s) f s in x_probs]
#plt.fill_between(lowfi(x.x, TRUE_SIGNALS[0][0]), label='Model')
#plt.plot(lowfi(x.x, TRUE_SIGNALS[0][0]), label='Model')
#plt.legend()
plt.savefig('Tuned.pdf')
