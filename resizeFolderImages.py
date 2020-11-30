import PIL
import os
from PIL import Image

def resizeFolder():
    f = r'C:\Users\robin\Desktop\charlieNew\deepCharlie\imgtoresize'
    for file in os.listdir(f):
        f_img = f+"/"+file
        img = Image.open(f_img)
        img = img.resize((100,100))
        img.save(f_img)

if __name__ == "__main__":
    resizeFolder()