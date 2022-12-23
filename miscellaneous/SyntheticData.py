import starry
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
import exoplanet
from starry_process import StarryProcess
from scipy.linalg import cho_factor, cho_solve
import starry_process 
import theano
import aesara_theano_fallback.tensor as tt
import aesara.tensor as at
from theano.tensor.slinalg import cholesky

starry.config.quiet = True
np.random.seed(1)


# Planet orbit
p_inc = 88.0
p_ecc = 0.2
p_Omega = 30.0
p_w = 30.0
p_porb = 1.0
p_t0 = 0.5
p_r = 0.1

# Star
s_inc = 75
s_u = [0.4, 0.2]
s_prot = 4.3

# Gaussian process for the spots
gp_mu = 30.0
gp_sig = 5.0
gp_r = 10.0
gp_c = 0.1
gp_n = 10.0

# Time array (10 days @ 1 min cadence)
t = np.arange(0, 10, 1 / 24 / 60)

# Get the Cartesian position of the planet
star = starry.Primary(starry.Map())
planet = starry.Secondary(
    starry.Map(),
    inc=p_inc,
    ecc=p_ecc,
    Omega=p_Omega,
    w=p_w,
    porb=p_porb,
    t0=p_t0
)
sys = starry.System(star, planet)
xo, yo, zo = sys.position(t)
xo = xo.eval()[1]
yo = yo.eval()[1]
zo = zo.eval()[1]

# Get the flux design matrix
map = starry.Map(15, len(s_u))
map.inc = s_inc
for n, coeff in enumerate(s_u):
    map[n + 1] = coeff
theta = (360 * t / s_prot) % 360
A = map.design_matrix(theta=theta, xo=xo, yo=yo, zo=zo, ro=p_r).eval()

# Draw 10 samples from the GP
sp = StarryProcess(mu=gp_mu, sigma=gp_sig, r=gp_r, c=gp_c, n=gp_n)
y = sp.sample_ylm(nsamples=10).eval().T

# Starry process Ylms are normalized so that they have
# zero luminosity (i.e., a featureless star has Y_{0,0} = 0)
# Let's renormalize it to unity, since that's the baseline
# we want for transits
y[0] += 1

# Compute the light curves
flux0 = (A @ y).T

error = 1e-3
err = np.ones_like(flux0) * error
flux = np.array(flux0)
flux += np.random.randn(len(t)) * err


starry.config.lazy = True
with pm.Model() as model:
    # Params for the flux model
    # Orbital parameters for the planet.
    porb = pm.Normal("porb", mu=1.0, sigma=0.001)
    t0 = pm.Normal("t0", mu=0.5, sigma=0.01)
    u1 = pm.Uniform("u1", lower=0.39, upper=0.41)
    u2 = pm.Uniform("u2", lower=0.19, upper=0.21)
    rp = pm.Uniform("rp", lower=0.09, upper=0.11)

    # Instantiate the star; all its parameters are assumed
    # to be known exactly, ecept for the limb-darkening coefficients
    star = starry.Primary(starry.Map())
    planet = starry.Secondary(
        starry.Map(),
        inc=p_inc,
        ecc=p_ecc,
        Omega=p_Omega,
        w=p_w,
        porb=p_porb,
        t0=p_t0
    )
    sys = starry.System(star, planet)
    xo, yo, zo = sys.position(t)
    xo = xo.eval()[1]
    yo = yo.eval()[1]
    zo = zo.eval()[1]

    # Get the flux design matrix
    map = starry.Map(15, len(s_u))
    map.inc = s_inc
    for n, coeff in enumerate(s_u):
        map[n + 1] = coeff
    theta = (360 * t / s_prot) % 360
    A = map.design_matrix(theta=theta, xo=xo, yo=yo, zo=zo, ro=p_r)

    mean = pm.Normal("mean", mu=0.0, sd=10, testval=0)

    # flux_model = pm.Deterministic(
    #         "flux_model", (sys.flux(t)-1)*1e3
    #     )

    # Things we know
    u = [u1,u2]
    ferr = 1e-3

    # Spot latitude params. Isotropic prior on the mode
    # and uniform prior on the standard deviation
    unif0 = pm.Uniform("unif0", 0.0, 1.0)
    mu = 90 - tt.arccos(unif0) * 180 / np.pi
    pm.Deterministic("mu", mu)
    sigma = pm.Uniform("sigma", 1.0, 20.0)

    # Spot radius (uniform prior)
    r = pm.Uniform("r", 10.0, 30.0)

    # Spot contrast & number of spots (uniform prior)
    c = pm.Uniform("c", 0.0, 0.5, testval=0.1)
    n = pm.Uniform("n", 1.0, 30.0, testval=5)

    # Inclination (isotropic prior)
    unif1 = pm.Uniform("unif1", 0.0, 1.0)
    i = tt.arccos(unif1) * 180 / np.pi
    pm.Deterministic("i", i)

    # Period (uniform prior)
    p = pm.Uniform("p", 0.75, 1.25)

    # Variability timescale (uniform prior)
    tau = pm.Uniform("tau", 0.1, 10.0)

    # # Instantiate the GP
    sp = StarryProcess(mu=mu, sigma=sigma, r=r, c=c, n=n)

    # Compute the log likelihood
    # Cov matrix at the ylm basis
    Sigma_ylm = sp.cov_ylm
    K = len(t)

    # Cov matrix
    Sigma = tt.dot(tt.dot(A,Sigma_ylm),A.T)
    Sigma+=1 # adding the baseline
    Sigma += err[0]**2*np.eye(len(t)) # add error to the covariance to make it positive definite
    cho_gp_cov = cholesky(Sigma)
    mean1 = tt.reshape(tt.dot(A,sp.mean_ylm + 1), (K, 1))
    r = (
        tt.reshape(tt.transpose(tt.as_tensor_variable(flux[0])), (K, -1)) # this is for the first light curve!
        - mean1
    )
    M = r.shape[1]
    lnlikemod = -0.5 * tt.sum(
            tt.batched_dot(
                tt.transpose(r), tt.transpose(theano.tensor.slinalg.solve_lower_triangular(tt.transpose(cho_gp_cov),theano.tensor.slinalg.solve_lower_triangular(cho_gp_cov,r)))
            )
        )
    lnlikemod -= M * tt.sum(tt.log(tt.diag(cho_gp_cov)))
    lnlikemod -= 0.5 * K * M * tt.log(2 * np.pi)

    pm.Potential("lnlike", lnlikemod)


with model:
    loglike = pmx.eval_in_model(model['lnlike'])

print("loglike is ", loglike)

# Optimize the MAP solution.
with model:
    map_soln = pmx.optimize()


print("optimized loglike is ", map_soln['lnlike'])

with model:
    trace = pmx.sample(
        tune=250,
        draws=500,
        start=map_soln,
        chains=4,
        cores=1,
        target_accept=0.9,
    )

import corner

samples = pm.trace_to_dataframe(trace, varnames=["tau", "p", "unif1", "n", "c", "r", "sigma", "unif0", "mean", "rp", "u2", "u1", "t0", "porb"])


import arviz as az
with model:
    _ = az.plot_trace(trace, var_names=["tau", "p", "unif1", "n", "c", "r", "sigma", "unif0", "mean", "rp", "u2", "u1", "t0", "porb"])
    plt.savefig('arvizfig')




