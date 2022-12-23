#conda environment: pymc3_env
import os
os.system('conda deactivate')
os.system("conda activate pymc3_env")


from starry_process import StarryProcess
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sp = StarryProcess()
for param in ["r", "dr", "a", "b", "mu", "sigma", "c", "n"]:
    print("{} = {}".format(param, getattr(sp, param).eval()))

#Sampling
#theano.config.compute_test_value = "warn"
y = sp.sample_ylm().eval()

sp.visualize(y, colorbar=True)

t = np.linspace(0, 4, 250)
flux = sp.flux(y, t).eval()

plt.plot(t, 1e3 * flux[0])
plt.xlabel("rotations")
plt.ylabel("relative flux [ppt]")
plt.show()

flux = sp.sample(t, nsamples=50, eps=1e-8).eval()
'''
for k in range(50):
    plt.plot(t, 1e3 * flux[k], alpha=0.5)
plt.xlabel("rotations")
plt.ylabel("relative flux [ppt]")
plt.show()
'''

#INFERENCE

#creating the noise:
ferr = 1e-3
np.random.seed(0)
f = flux + ferr * np.random.randn(50, len(t))
'''
plt.plot(t, f[0], "C0.", ms=3)
plt.xlabel("rotations")
plt.ylabel("relative flux [ppt]")
plt.show()
'''

#Compiling theano functions
import theano
import theano.tensor as tt

f_tensor = tt.dvector()
r_tensor = tt.dscalar()

sp = StarryProcess(r=r_tensor)
log_likelihood_tensor = sp.log_likelihood(t, f_tensor, ferr ** 2)

log_likelihood = theano.function([f_tensor, r_tensor], log_likelihood_tensor)


r = np.linspace(10, 45, 100)
ll = np.zeros_like(r)
for k in tqdm(range(len(r))):
    ll[k] = log_likelihood(f[0], r[k])


likelihood = np.exp(ll - np.max(ll))
plt.plot(r, likelihood, label="likelihood")
plt.axvline(20, color="C1", label="truth")
plt.legend()
plt.ylabel("relative likelihood")
plt.xlabel("spot radius [degrees]")
plt.show()


prob = likelihood / np.trapz(likelihood, r)
plt.plot(r, prob, label="posterior")
plt.axvline(20, color="C1", label="truth")
plt.legend()
plt.ylabel("probability density")
plt.xlabel("spot radius [degrees]")
plt.show()

#ENSEMBLE ANALYSES

r = np.linspace(10, 45, 100)
ll = np.zeros_like(r)
for k in tqdm(range(len(r))):
    ll[k] = np.sum([log_likelihood(f[n], r[k]) for n in range(50)])

likelihood = np.exp(ll - np.max(ll))
prob = likelihood / np.trapz(likelihood, r)
plt.plot(r, prob, label="posterior")
plt.axvline(20, color="C1", label="truth")
plt.legend()
plt.ylabel("probability density")
plt.xlabel("spot radius [degrees]")
plt.show()



