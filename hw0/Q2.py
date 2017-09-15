from PIL import Image
import sys
path = sys.argv[1]

with open(path) as inputimage:
    pixel = inputimage.load()

    for i in range(inputimage.size[0]):
        for j in range(inputimage.size[1]):
            pixel[i,j] = (pixel[i,j][0]//2,pixel[i,j][1]//2,pixel[i,j][2]//2)
            
    inputimage.save("Q2.png")
