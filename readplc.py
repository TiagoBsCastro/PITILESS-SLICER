import ReadPinocchio as rp
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

plc = rp.plc("pinocchio.cdm.plc.out")
cat = rp.catalog("pinocchio.0.0000.cdm.catalog.out")
hist = rp.histories("pinocchio.cdm.histories.out")
