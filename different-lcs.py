import numpy as np
import starry
import theano
theano.config.gcc__cxxflags += " -fexceptions"
theano.config.on_opt_error = "raise"
theano.tensor.opt.constant_folding
theano.graph.opt.EquilibriumOptimizer
import matplotlib.pyplot as plt

starry.config.lazy = False
starry.config.quiet = True

t = np.linspace(0, 1, 500)

map0 = starry.Map(ydeg=1)

theta=np.linspace(0, 360, 500)

flux0 = map0.flux(theta=theta)

plt.plot(t, flux0, 'k-')
plt.xlabel("time [days]")
plt.ylabel("normalized flux");
plt.show()

map0.show(theta=np.linspace(0, 360, 50),grid=False, file='spotless.gif')

map = starry.Map(ydeg=30, udeg=2, amp=1.0)
star = starry.Primary(map, m=1.0, r=1.0, prot=1.0)
star.map[1] = 0.40
star.map[2] = 0.26

planet = starry.kepler.Secondary(
    starry.Map(ydeg=5, amp=5e-3),  # the surface map
    m=0,  # mass in solar masses
    r=0.1,  # radius in solar radii
    porb=1.0,  # orbital period in days
    ecc=0.,  # eccentricity
    t0=0.5,  # time of transit in days
)

system = starry.System(star, planet)

flux_star, flux_planet = system.flux(t, total=False)

plt.plot(t, flux_star, 'k-')
plt.xlabel("time [days]")
plt.ylabel("normalized flux");
plt.show()
system.show(t=np.linspace(0, 1, 50), window_pad=4, figsize=(8, 8), file='planet-star.gif')

contrast = 0.25
radius = 15
map.spot(contrast=contrast, radius=radius)
map.show(theta=np.linspace(0, 360, 50),grid=False)

star = starry.Primary(map, m=1.0, r=1.0, prot=10.0)
star.map[1] = 0.40
star.map[2] = 0.26

planet = starry.kepler.Secondary(
    starry.Map(ydeg=5, amp=5e-3),  # the surface map
    m=0,  # mass in solar masses
    r=0.1,  # radius in solar radii
    porb=0.6,  # orbital period in days
    ecc=0.,  # eccentricity
    t0=0.5,  # time of transit in days
)
system = starry.System(star, planet)

flux_star, flux_planet = system.flux(t, total=False)

plt.plot(t, flux_star, 'k-')
plt.xlabel("time [days]")
plt.ylabel("normalized flux");
plt.show()

system.show(t=np.linspace(0, 1, 50), window_pad=4, figsize=(8, 8), file='planet-star-one-spot.gif')



star = starry.Primary(map, m=1.0, r=1.0, prot=2.0)
star.map[1] = 0.40
star.map[2] = 0.26
star.map[1:, :] = 0.01 * np.random.randn(star.map.Ny - 1)

planet = starry.kepler.Secondary(
    starry.Map(ydeg=5, amp=5e-3),  # the surface map
    m=0,  # mass in solar masses
    r=0.1,  # radius in solar radii
    porb=0.6,  # orbital period in days
    ecc=0.,  # eccentricity
    t0=0.5,  # time of transit in days
)
system = starry.System(star, planet)

flux_star, flux_planet = system.flux(t, total=False)

plt.plot(t, flux_star, 'k-')
plt.xlabel("time [days]")
plt.ylabel("normalized flux");
plt.show()

system.show(t=np.linspace(0, 1, 50), window_pad=4, figsize=(8, 8), file='planet-star-many-spots.gif')

