import re
import numpy as np

def getValue (s,string,type):

	try:
		return type(re.search(s+'(.*)', string).group(1))
	except :
		print "Unknown variable:",s
		raise TypeError

intArrayFromString = lambda s: np.fromstring(s, dtype=int, sep=' ')

def ImportParams (log):

	s = open(log,"r").read()
	result = re.search('PARAMETER VALUES from file LargePLC_FS_[0-9].params:\n(.*)GENIC parameters:\n', s,flags=re.DOTALL).group(1)

	Omega0 = getValue("Omega0",result,float)
	OmegaLambda = getValue("OmegaLambda",result,float)
	Hubble100 = getValue("Hubble100",result,float)
	GridSize = getValue("GridSize",result,intArrayFromString)
	BoxSizeTrueMpc = getValue("BoxSize \(true Mpc\)",result,float)
	BoxSize = getValue("BoxSize \(Mpc\/h\)",result,float)
	ParticleMassTrueMsun = getValue("Particle Mass \(true Msun\)",result,float)
	ParticleMass = getValue("Particle Mass \(Msun\/h\)",result,float)
	MinHaloMass = getValue("MinHaloMass \(particles\)",result,int)
	NumFiles = getValue("NumFiles",result,int)

	print "\n           Values Read from Log file         \n"
	print 'Omega0',Omega0
	print 'OmegaLambda:', OmegaLambda
	print 'Hubble100:', Hubble100
	print 'GridSize:', GridSize
	print 'BoxSizeTrueMpc:', BoxSizeTrueMpc
	print 'BoxSize:', BoxSize
	print 'ParticleMassTrueMsun:', ParticleMassTrueMsun
	print 'ParticleMass:', ParticleMass
	print 'MinHaloMass:', MinHaloMass
	print 'NumFiles:', NumFiles

        fov = float(re.search('It will have an aperture of (.*) degrees', s).group(1))

	print 'Angular Aperture of PLC:', fov

	return Omega0,OmegaLambda,Hubble100,GridSize,BoxSizeTrueMpc,BoxSize,ParticleMassTrueMsun,ParticleMassTrueMsun,ParticleMass,MinHaloMass,NumFiles,fov
