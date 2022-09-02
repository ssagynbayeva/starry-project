# conda environment: pymc3_env

import starry
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
import lightkurve as lk
import aesara_theano_fallback.tensor as tt
import pandas as pd
from celerite2.theano import terms, GaussianProcess
import theano

# Disable warnings
theano.tensor.opt._logger.setLevel(50)
starry.config.quiet = True

search_result = lk.search_lightcurve("HAT-P-11", author="Kepler", cadence="short")

# dowloading only the 3rd quarter data
lc = search_result[5:8].download_all()

for i in range(len(lc)):
    lc[i] = lc[i].remove_nans()
    lc[i] = lc[i].remove_outliers()
    lc[i] = lc[i][lc[i].quality == 0] # enable only 0 quality flags

def gp_model(lightcurve):
    starry.config.lazy = True
    with pm.Model() as model:

        # Orbital parameters for the planet.
        porb = pm.Normal("porb", mu=4.888, sigma=0.001)
        t0 = pm.Normal("t0", mu=261.665, sigma=0.01)
        u1 = pm.Uniform("u1", lower=0.638, upper=0.7)
        u2 = pm.Uniform("u2", lower=0.033, upper=0.05)
        rp = pm.Uniform("rp", lower=0.03232321, upper=0.04632)

        # Instantiate the star; all its parameters are assumed
        # to be known exactly, ecept for the limb-darkening coefficients
        A = starry.Primary(
            starry.Map(ydeg=0, udeg=2, amp=1.0), m=0.809, r=0.683, prot=1.0
        )
        A.map[1] = u1
        A.map[2] = u2

        # Instantiate the planet. Everything is fixed except for
        # its luminosity and the hot spot offset.
        b = starry.Secondary(
            starry.Map(ydeg=1, udeg=0, amp=0, obl=0.0),
            m=7.0257708e-5,  # mass in solar masses
            r=rp,  # radius in solar radii
            inc=88.99,  # orbital inclination
            porb=porb,  # orbital period in days
            prot=1,
            w=-162.149,  # Argument of periastron (little omega)
            ecc=0.265,  # eccentricity
            Omega=106,
            t0=t0,
        )

        # A term to describe the non-periodic variability
        sigma = pm.InverseGamma(
            "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        rho = pm.InverseGamma("rho", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0))

        # Instantiate the system as before
        sys = starry.System(A, b)

        for l in range(len(lightcurve)):

            x = lightcurve[l].time.value
            y = lightcurve[l].flux.value
            yerr = lightcurve[l].flux_err.value
            mu = np.median(y)
            y = (y / mu - 1) * 1e3
            yerr = yerr * 1e3 / mu

            # plt.figure(figsize=(20, 8))
            # plt.plot(x,y, "k.", ms=2, alpha=0.5);

            mean = pm.Normal(f"mean_{l}", mu=0.0, sd=10, testval=0)

            # Our model for the flux in ppt
            flux_model = pm.Deterministic(
                f"flux_model_{l}", (sys.flux(x) - 1) * 1e3
            )

            # Use the GP model from the stellar variability tutorial

            # A jitter term describing excess white noise
            log_jitter = pm.Normal(
                f"log_jitter_{l}", mu=np.log(np.mean(yerr)), sigma=2.0
            )

            # Set up the Gaussian Process model
            kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1 / 3.0)
            gp = GaussianProcess(
                kernel,
                t=x,
                diag=yerr**2 + tt.exp(2 * log_jitter),
                mean=mean,
                quiet=True,
            )

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            gp.marginal(f"gp_{l}", observed=y - flux_model)

            # Compute the GP model prediction for plotting purposes
            gp_pred = pm.Deterministic(f"gp_pred_{l}", gp.predict(y - flux_model))

    return model

def plot_whole_model(lightcurve,model,point=None):
    if point is None:
        point = model.test_point
    x = []
    y = []
    yerr = []
    flux_model = []
    gp_pred = []
    
    with model:

        for i in range(len(lightcurve)):
            flux_model.append(pmx.eval_in_model(model[f'flux_model_{i}']))
            gp_pred.append(pmx.eval_in_model(model[f'gp_pred_{i}']))

            mu = np.median(lightcurve[i].flux)
            lightcurve[i].flux = (lightcurve[i].flux / mu - 1) * 1e3
            lightcurve[i].flux_err = lightcurve[i].flux_err * 1e3 / mu
            y.append(lightcurve[i].flux.value)
            x.append(lightcurve[i].time.value)
            yerr.append(lightcurve[i].flux_err.value)

        x = np.concatenate((x))
        y = np.concatenate((y))
        yerr = np.concatenate((yerr))

        gp_mod = np.concatenate((gp_pred))
        transit_mod = np.concatenate((flux_model)) 

        plt.figure(figsize=(20,8))
        plt.plot(x, y, 'k.')
        plt.plot(x, transit_mod, "C0-", lw=1)
        plt.plot(x, gp_mod, "C1-", lw=1)
        plt.show()

def plot_transit_fit(lightcurve,model,point=None):
    if point is None:
        point = model.test_point
    flux_model = []
    gp_pred = []
    x = []
    y = []
    yerr = []

    with model:

        for i in range(len(lightcurve)):
            flux_model.append(pmx.eval_in_model(model[f'flux_model_{i}']))
            gp_pred.append(pmx.eval_in_model(model[f'gp_pred_{i}']))

            mu = np.median(lightcurve[i].flux)
            lightcurve[i].flux = (lightcurve[i].flux / mu - 1) * 1e3
            lightcurve[i].flux_err = lightcurve[i].flux_err * 1e3 / mu
            y.append(lightcurve[i].flux.value)
            x.append(lightcurve[i].time.value)
            yerr.append(lightcurve[i].flux_err.value)

        gp_mod = np.concatenate((gp_pred))
        transit_mod = np.concatenate((flux_model))
            
        x = np.concatenate((x))
        y = np.concatenate((y))
        yerr = np.concatenate((yerr))


        plt.figure(figsize=(20,8))
        plt.plot(x, y - gp_mod, 'k.')
        plt.plot(x, transit_mod, "C0-", lw=1)
        plt.xlim(310,312)
        plt.show()

def plot_transits(lightcurve, model, point=None):
    if point is None:
        point = model.test_point
    flux_model = []
    gp_pred = []
    x = []
    y = []
    yerr = []
    with model:

        for i in range(len(lightcurve)):
            flux_model.append(pmx.eval_in_model(model[f'flux_model_{i}'],point=model.test_point))
            gp_pred.append(pmx.eval_in_model(model[f'gp_pred_{i}']))

        fig, ax = plt.subplots(16, 2, figsize=(14, 16), sharex=True)

        # Normalize the data
        for i in range(len(lightcurve)):
            mu = np.median(lightcurve[i].flux)
            lightcurve[i].flux = (lightcurve[i].flux / mu - 1) * 1e3
            lightcurve[i].flux_err = lightcurve[i].flux_err * 1e3 / mu
            y.append(lightcurve[i].flux.value)
            x.append(lightcurve[i].time.value)
            yerr.append(lightcurve[i].flux_err.value)

        x = np.concatenate((x))
        y = np.concatenate((y))
        yerr = np.concatenate((yerr))

        print(x)

        for n in range(16):
            dt = 0.15
            tn = pmx.eval_in_model(model.t0, point=point) + n * pmx.eval_in_model(model.porb, point=point)
            idx = (x > tn - dt) & (x < tn + dt)
            ax[n, 0].plot(x[idx] - tn, y[idx], "k.", ms=2, alpha=0.3)

            tran_model = np.concatenate(flux_model)[idx]
            
            gp_model = np.concatenate(gp_pred)[idx]
            
            tran_model_off = tran_model + gp_model - tran_model

            ax[n, 0].plot(x[idx] - tn, tran_model_off, "C0-", label=f"light curve {i}")
            ax[n, 0].plot(x[idx] - tn, gp_model, "C1-", label=f"light curve {i}")

            ax[n, 1].plot(x[idx] - tn, y[idx], "k.", ms=2, alpha=0.5, label=f"light curve {i}")
            ax[n, 1].plot(x[idx] - tn, tran_model + gp_model, "C2-", label=f"light curve {i}");
        plt.legend()
        plt.show()

model = gp_model(lc)


# plot_whole_model(lc,model)
# plot_transits(lc, model)
plot_transit_fit(lc,model)


# Optimize the MAP solution.
# with model:
#     map_soln = pmx.optimize()

# plot_transits(lc, model,point=map_soln)


