from BlowFish_FacialRecognition import encrypt
from BlowFish_FacialRecognition import decrypt
from BlowFish_FacialRecognition import run

resolutions = ['500x500', '1000x1000', '2000x2000']
extentions = ['.png', '.tif', '.bmp', '.jpg']

enc_path = "..\\Headshot\\"+resolutions[0]+extentions[2]
dec_path = "..\\Headshot\\"+resolutions[0]+extentions[1]

run(enc_path, dec_path)
print()
encrypt(enc_path, "plain.txt")
decrypt(dec_path, "cipher.txt", "nonce.txt")

