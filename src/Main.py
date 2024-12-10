from BlowFish_FacialRecognition import encrypt
from BlowFish_FacialRecognition import decrypt
from BlowFish_FacialRecognition import run
from EuclidianDistances import run_comparison

path = 'Faces'
gender = ['M', 'F']
resolutions = ['512', '1024', '2048']
extentions = ['.png', '.tif', '.bmp', '.jpg']

enc_path = f"..\\{path}\\{gender[0]}{resolutions[2]}{extentions[0]}"
dec_path = f"..\\{path}\\{gender[0]}{resolutions[1]}{extentions[0]}"

run(enc_path, dec_path)
print()
encrypt(enc_path, "plain.txt")
decrypt(dec_path, "cipher.txt", "nonce.txt")


