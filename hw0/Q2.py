from PIL import Image
import sys
path = sys.argv[1]

image = Image.open(path)
pixel = image.load()

for i in range(image.size[0]):
    for j in range(image.size[1]):
        pixel[i,j] = (pixel[i,j][0]//2,pixel[i,j][1]//2,pixel[i,j][2]//2)
        
image.save("Q2.png")
