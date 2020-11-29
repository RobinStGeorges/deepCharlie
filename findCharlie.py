from PIL import Image
import os

def crop(infile,height,width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield im.crop(box)

if __name__=='__main__':
    infile="image3.jpg"
    height=100
    width=100
    start_num=0
    for k,piece in enumerate(crop(infile,height,width),start_num):
        img=Image.new('RGB', (height,width), 255)
        img.paste(piece)
        # path=os.path.join('\\tmp\\IMG-' + str(k) + '.jpg')
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        img.save(THIS_FOLDER + '/tmp/IMG-' + str(k) + '.jpg')