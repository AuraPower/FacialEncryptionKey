import cv2 as cv # type: ignore
import numpy as np
from Crypto.Cipher import Blowfish
from Crypto.Random import get_random_bytes
from Crypto.Util import Counter
import hashlib
import dlib

def magnitude(points):
    """
    Returns the rounded magnitudes of an array of postions

    Args:
        points: list of tuples of x,y coordinates

    Return: 
        np array of int magnitudes 
    """

    # Convert the list of tuples to a NumPy array for vectorized operations
    points_array = np.array(points)
    
    # Calculate magnitudes using the formula: sqrt(x^2 + y^2)
    magnitudes = np.sqrt(np.sum(points_array**2, axis=1))

    for i in range(len(magnitudes)):
        magnitudes[i] = round(magnitudes[i])

    return magnitudes

def encrypt_blowfish_ctr(key, plaintext):
    """
    Encrypts plaintext using Blowfish in CTR mode.

    Args:
        key: The encryption key (bytes).
        plaintext: The data to encrypt (bytes).

    Returns:
        A tuple containing the ciphertext (bytes) and the nonce (bytes).
    """

    # Generate a nonce (Initialization Vector)
    nonce = get_random_bytes(8)  # Blowfish block size is 8 bytes

    ctr = Counter.new(64, initial_value=int.from_bytes(nonce, byteorder='big'))

    cipher = Blowfish.new(key, Blowfish.MODE_CTR, counter=ctr)

    # Get ciphertext from plaintext
    ciphertext = cipher.encrypt(plaintext)

    return ciphertext, nonce

def decrypt_blowfish_ctr(key, ciphertext, nonce):
    """
    Decrypts ciphertext encrypted with Blowfish in CTR mode.

    Args:
        key: The encryption key (bytes).
        ciphertext: The data to decrypt (bytes).
        nonce: The nonce used for encryption (bytes).

    Returns:
        The decrypted plaintext (bytes).
    """
    ctr = Counter.new(64, initial_value=int.from_bytes(nonce, byteorder='big'))
    cipher = Blowfish.new(key, Blowfish.MODE_CTR, counter=ctr)
    plaintext = cipher.decrypt(ciphertext)
    return plaintext

def process_image(path):
    """
    Proccesses image for use in feature generation

    Args:
        path: file path to image file on local device
    
    Return:
        Returns image of resolution size 256x256 px

    """
    img = cv.imread(path)
    if img is None:
        raise ValueError(f"Error: Could not open or read image: {path}. Please check the file path.")

    img = identify_face(img)
    img = cv.resize(img, (256, 256))

    return img

def visualize_feature_points(img, feature_points):
    for (x, y) in feature_points:
        cv.circle(img, (x, y), 2, (0, 255, 0), -1)  # Draw green dots for landmarks
    cv.imshow("Feature Points", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def generate_feature_points(img):

    landmark_model_path = "..\\Models\\shape_predictor_68_face_landmarks.dat"

    # Load the dlib face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_model_path)

    img = cv.resize(img, (256, 256))

    faces = detector(img)
    if len(faces) == 0:
        print("No faces detected in the image.")
        return None
    
    face = faces[0]
    landmarks = predictor(img, face)  # Get the landmarks

    # Extract x, y coordinates of the feature points
    feature_points = []
    for i in range(68):  # dlib provides 68 landmark points
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        feature_points.append((x, y))

    return feature_points

def identify_face(img):
    """
    Identifies faces in image using OpenCV classifier

    Args:
        img: image type from cv.imread 

    Returns:
        A cropped image of the face detected
    """

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    if len(faces) == 0:
        raise RuntimeError("No face detected")
        
    # Take the first detected face (or iterate over all faces if needed)
    x, y, w, h = faces[0]  # Coordinates of the first face
    face_roi = img[y:y+h, x:x+w]

    return face_roi

def generate_key(feature_points):

    # Flatten the feature points and convert them to a byte array
    feature_array = np.array(feature_points, dtype=np.int32).flatten()
    feature_bytes = feature_array.tobytes()

    # Hash the feature bytes to generate a key
    key = hashlib.sha256(feature_bytes).digest()[:56]  # Generate 56-byte key

    return key

############### IO ENC DEC FUNCTIONS #################

def encrypt(img_path, plain_path):
    """
    Encrypts a plaintext file using an encryption key derived from an image.

    Args:
        img_path: Path to the image used for generating the key.
        plain_path: Path to the plaintext file to be encrypted.

    Saves:
        - Encrypted ciphertext to 'cipher.txt'.
        - Generated nonce to 'nonce.txt'.
    """
    # Read plain text as binary
    with open(plain_path, "rb") as file:
        plain_text = file.read()

    # generate key from feature points. generate feature points from image
    img = process_image(img_path)
    feature_points = generate_feature_points(img)
    key = generate_key(feature_points)

    ciphertxt, noncetxt = encrypt_blowfish_ctr(key, plain_text)

    # write to files
    with open("cipher.txt", "wb") as cipher:
        cipher.write(ciphertxt)

    with open("nonce.txt", "wb") as nonce:
        nonce.write(noncetxt)

def decrypt(img_path, cipher_path, nonce_path):
    """
    Decrypts a ciphertext file using an encryption key derived from an image.

    Args:
        img_path: Path to the image used for generating the key.
        cipher_path: Path to the file containing the encrypted ciphertext.
        nonce_path: Path to the file containing the nonce.

    Returns:
        The decrypted plaintext as a string.
    """
    # Read the cipher text as binary
    with open(cipher_path, "rb") as file:
        cipher_text = file.read()

    # Read the nonce as binary
    with open(nonce_path, "rb") as file:
        nonce = file.read()  

    # generate key from feature points. generate feature points from image
    img = process_image(img_path)
    feature_points = generate_feature_points(img)
    key = generate_key(feature_points)

    # Decrypt the ciphertext
    decrypted_text = decrypt_blowfish_ctr(key, cipher_text, nonce)

    # Convert bytes to string
    try:
        
        with open("decrypted.txt", "w") as decrypted:
            decrypted.write(decrypted_text.decode())
        
    except:
        with open("decrypted.txt", "w") as decrypted:
            decrypted.write(""+decrypted_text.hex())
        print("DECRYPTION FAILED")

############## RUN ##################
def run(enc_path, dec_path):
    enc_img = process_image(enc_path)
    dec_img = process_image(dec_path)

    enc_feature_points = generate_feature_points(enc_img)
    dec_feature_points = generate_feature_points(dec_img)

    visualize_feature_points(enc_img, enc_feature_points)
    visualize_feature_points(dec_img, dec_feature_points)

    enc_feature_point_magnitudes = magnitude(enc_feature_points)
    dec_feature_point_magnitudes = magnitude(dec_feature_points)

    print(f"Enc Point Magnitudes {enc_feature_point_magnitudes}")
    print()
    print(f"Dec Point Magnitudes {dec_feature_point_magnitudes}")
    print()
    print(f"Magnitude Difference: {enc_feature_point_magnitudes-dec_feature_point_magnitudes}")


    encryption_key = generate_key(enc_feature_point_magnitudes)
    decryption_key = generate_key(dec_feature_point_magnitudes)

    print(f"Encryption Key Hex: {encryption_key.hex()}")
    print(f"Decryption Key Hex: {decryption_key.hex()}")

    if (encryption_key) and (decryption_key):
        plaintext = b'This is a secret message.'

        ciphertext, nonce = encrypt_blowfish_ctr(encryption_key, plaintext)
        print(f"Plain text: {plaintext}")
        print(f"Ciphertext: {ciphertext.hex()}")

        decrypted_text = decrypt_blowfish_ctr(decryption_key, ciphertext, nonce)

        
        if (decrypted_text == plaintext):
            print("Decryption Successful")
            print(f"Decrypted text: {decrypted_text}")
            return True
        else:
            print("Decryption failed")
            print(f"Decrypted text: {decrypted_text}")
            return False
    else:
        print("Key generation failed.")

def test(enc_path, dec_path):
    img_path = '..\\Faces\\'

    enc_img = process_image(img_path+enc_path)
    dec_img = process_image(img_path+dec_path)

    enc_feature_points = generate_feature_points(enc_img)
    dec_feature_points = generate_feature_points(dec_img)

    enc_feature_point_magnitudes = magnitude(enc_feature_points)
    dec_feature_point_magnitudes = magnitude(dec_feature_points)

    encryption_key = generate_key(enc_feature_point_magnitudes)
    decryption_key = generate_key(dec_feature_point_magnitudes)


    if (encryption_key) and (decryption_key):
        plaintext = b'This is a secret message.'

        ciphertext, nonce = encrypt_blowfish_ctr(encryption_key, plaintext)

        decrypted_text = decrypt_blowfish_ctr(decryption_key, ciphertext, nonce)

        
        if (decrypted_text == plaintext):
            return True
        else:
            return False
    else:
        print("Key not generated")


if __name__ == "__main__":
    run()