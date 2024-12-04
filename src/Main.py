from BlowFish_FacialRecognition import run

resolutions = ['500x500', '1000x1000', '2000x2000']
extentions = ['.png', '.jpg', '.tif', '.bmp']

enc_path = resolutions[2]+extentions[2]
dec_path = resolutions[2]+extentions[3]

files = ['500x500.png','500x500.jpg','500x500.tif','500x500.bmp','1000x1000.png','1000x1000.jpg',
         '1000x1000.tif','1000x1000.bmp','2000x2000.png','2000x2000.jpg','2000x2000.tif','2000x2000.bmp']

for enc in files:
    for dec in files:
        run(enc, dec)

run(enc_path, dec_path)