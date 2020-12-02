import os
import shutil

import CNN as CNN
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import math
import glob

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
        # print('row size : ', rowSize)
        img = Image.open('tmp/IMG-' + str(i) + '.jpg')
        i += 1
        # print(element)
        if acc >= rowSize: #End of file row
            acc = 0
            accY += 1
            # print('acc : '+ str(acc) +' accY : '+ str(accY) )
            newImg.paste(img, (acc * 100, accY * 100))
        else:
            # print('acc : ' + str(acc) + ' accY : ' + str(accY))
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
    # print('##########################')
    # print('')
    if listData[1]>=2:
        print('Resultat : ', listData[1])
        drawRectangleOnPath(path)
        print('C\'est charlie !')
    # print('')
    # print('##########################')


if __name__=='__main__':
    filePath= "source3.jpg"
    originalWidth = (Image.open(filePath).width)
    originalHeigth = (Image.open(filePath).height)
    # print('originalWidth : ', originalWidth)
    # print('originalHeigth : ', originalHeigth)
    height=100
    width=100
    start_num=0

    # Decoupage de l'image en taille 100*100
    os.mkdir(os.path.join("tmp"))
    for k,piece in enumerate(crop(filePath,height,width),start_num):
        img=Image.new('RGB', (height,width), 255)
        img.paste(piece)
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        img.save(THIS_FOLDER + '/tmp/IMG-' + str(k) + '.jpg')

    # Creation du CNN a partir de la sauvegarde
    cnn = CNN.CNN() #charlie is in 95
    cnn.load_state_dict(torch.load("toto.dat"))
    cnn.eval() #toujours commencer par ça pour construire le réseau

    # Recherche de charlie dans chacune des images de tmp
    for element in os.listdir('tmp/'):
        imgPath = element
        tensoredImg = imgPathToTensor('tmp/' + imgPath)
        output = cnn(tensoredImg)
        checkIsCharlie(output, 'tmp/' + imgPath)

    # Recreation de l'image à partir des elements dans tmp/
    join('tmp/', math.floor(abs(originalWidth/100))-1, originalWidth, originalHeigth)

    # Supression des images 100*100
    shutil.rmtree(os.path.join("tmp"))


