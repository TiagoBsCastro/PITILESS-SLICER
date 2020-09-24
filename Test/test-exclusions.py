import re

for i,j in zip([1], [2]):

    print("\n####################################\n")

    for n in range(4):

        log = open("../box_log_{}.txt".format(n)).read()
        particles = re.findall(r".Number of Particles in this rank\:\s+(\d*)", log)

        if n == 0:

            print(n, int(particles[i])-int(particles[j]))

        elif n == 1:

            excluded = int(particles[i])-int(particles[j])
            print(n, excluded)

        else:

            excludedi = int(particles[i])-int(particles[j])

            if excluded == excludedi:
        
                print(n, excludedi)

            else:

                print(n, excludedi, excluded)
    
