import creat_Plc
import matplotlib.pyplot as plt
from imageio import imread, mimsave

pngs = []
images = []
for zet in 1.0/np.linspace(1.0/21, 1.0) - 1.0:


   filename = "z_{}.png".format(round(zet,2))
   xx, yy, zz = snapPos(zet)
   cut = zz < 10.0

   plt.scatter(xx[cut][::10], yy[cut][::10], label=str(round(zet,2)), s=0.01)
   plt.legend(loc=1)
   plt.savefig(filename)
   plt.close()
   pngs.append(filename)

for png in pngs:

    img = imread(png)
    images.append(img)

mimsave('movie.gif', images, duration = 0.1)

os.system('rm *png')

