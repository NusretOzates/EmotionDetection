import os

from PIL import Image

for infile in os.listdir("./"):
    print ("file : " + infile)
    if infile[-4:] == "tiff" or infile[-3:] == "bmp" :
       # print "is tif or bmp"
       outfile = infile[:-4] + "jpeg"
       im = Image.open(infile)
       print ("new filename : " + outfile)
       out = im.convert("RGB")
       out.save(outfile, "JPEG", quality=90)
