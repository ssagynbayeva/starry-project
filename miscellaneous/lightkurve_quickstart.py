from lightkurve import search_targetpixelfile
import numpy as np
import matplotlib.pyplot as plt

pixelfile = search_targetpixelfile("HAT-P-11").download_all()

kepler_data = pixelfile[:52]
kepler_data[0].plot(frame=1)


