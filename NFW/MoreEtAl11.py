from colossus.cosmology import cosmology
from colossus.halo import concentration

cosmology.setCosmology('planck15')
cvir = concentration.concentration(1E12, 'vir', 0.0, model = 'bullock01')
