import matplotlib
from PIL import Image
import os
from matplotlib import patches
import CNN as CNN
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

def crop(infile,height,width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield im.crop(box)

def join(folderPath, rowSize, originalWidth, originalHeigth):
    i = 0
    acc = 0
    accY = 0
    newImg = Image.new(mode = "RGB", size = (int(originalWidth) , int(originalHeigth) ))
    for element in os.listdir(folderPath):
        img = Image.open('tmp/IMG-' + str(i) + '.jpg')
        i += 1
        if acc > rowSize: #End of file row
            acc = 0
            accY += 1
            newImg.paste(img, (acc * 100, accY * 100))
        else:
            newImg.paste(img, (acc * 100, accY * 100))
            acc += 1


    newImg.save('result.jpg')

def imgPathToTensor(path):
    pil_img = Image.open(path)
    tensoredImg = transforms.ToTensor()(pil_img).unsqueeze_(0)
    return(tensoredImg)

def drawRectangleOnPath(path):
    source_img = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    draw.rectangle(((0, 00), (100, 100)), outline="red", fill=None, width=10)
    source_img = source_img.convert("RGB")
    source_img.save(path)

def checkIsCharlie(output, path):
    listData = output.data.tolist()[0]
    if listData[1]>2.35:
        drawRectangleOnPath(path)
        print('C\'est charlie !')

if __name__=='__main__':
    filePath= "source.jpg"
    originalWidth = Image.open(filePath).width-101
    originalHeigth = Image.open(filePath).height
    height=100
    width=100
    start_num=0
    for k,piece in enumerate(crop(filePath,height,width),start_num):
        img=Image.new('RGB', (height,width), 255)
        img.paste(piece)
        # path=os.path.join('\\tmp\\IMG-' + str(k) + '.jpg')
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        img.save(THIS_FOLDER + '/tmp/IMG-' + str(k) + '.jpg')

    cnn = CNN.CNN() #charlie is in 95
    cnn.load_state_dict(torch.load("toto.dat"))
    cnn.eval() #toujours commencer par ça pour construire le réseau
    for element in os.listdir('tmp/'):
        imgPath = element
        tensoredImg = imgPathToTensor('tmp/' + imgPath)
        output = cnn(tensoredImg)
        checkIsCharlie(output, 'tmp/' + imgPath)
    join('tmp/', abs(originalWidth/height), originalWidth, originalHeigth)


