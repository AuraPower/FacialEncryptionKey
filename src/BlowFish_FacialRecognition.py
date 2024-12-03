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

def generate_feature_points(img, landmark_model_path):

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
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(256, 256))
    if len(faces) == 0:
        print("No faces detected in the image.")
        return None

    # Take the first detected face (or iterate over all faces if needed)
    x, y, w, h = faces[0]  # Coordinates of the first face
    face_roi = img[y:y+h, x:x+w]  # Extract the region of interest (ROI)

    return face_roi

def generate_key(feature_points):

    # Flatten the feature points and convert them to a byte array
    feature_array = np.array(feature_points, dtype=np.int32).flatten()
    feature_bytes = feature_array.tobytes()

    # Hash the feature bytes to generate a key
    key = hashlib.sha256(feature_bytes).digest()[:16]  # Generate 16-byte key

    return key

############## RUN ##################
img_path = 'C:\\Users\\Owner\\Pictures\\Headshot\\'
enc_image_path = "1000x1000.png"
dec_image_path = "1000x1000 - Copy.png"
landmark_model_path = "shape_predictor_68_face_landmarks.dat"

enc_img = process_image(img_path+enc_image_path)
dec_img = process_image(img_path+dec_image_path)

enc_feature_points = generate_feature_points(enc_img, landmark_model_path)
dec_feature_points = generate_feature_points(dec_img, landmark_model_path)

visualize_feature_points(enc_img, enc_feature_points)
visualize_feature_points(dec_img, dec_feature_points)

enc_feature_point_magnitudes = magnitude(enc_feature_points)
dec_feature_point_magnitudes = magnitude(dec_feature_points)

print(f"Enc: {enc_feature_point_magnitudes}")
print(f"Dec: {dec_feature_point_magnitudes}")

encryption_key = generate_key(enc_feature_point_magnitudes)
decryption_key = generate_key(dec_feature_point_magnitudes)

print(f"Encryption Key Hex: {encryption_key.hex()}")
print(f"Decryption Key Hex: {decryption_key.hex()}")

if (encryption_key) and (decryption_key):
    plaintext = b'This is a secret message.'

    ciphertext, nonce = encrypt_blowfish_ctr(encryption_key, plaintext)
    print(f"Ciphertext: {ciphertext.hex()}")

    decrypted_text = decrypt_blowfish_ctr(decryption_key, ciphertext, nonce)

    if (decrypted_text == plaintext):
        print("Decryption Successful")
        print(f"Decrypted text: {decrypted_text}")
    else:
        print("Decryption failed")
else:
    print("Key generation failed.")

"""
encryption_key = generate_key_from_face(path+'500x500.png')

decryption_key = generate_key_from_face(path+'500x500.png')

print(f"Encryption Key Hex: {encryption_key.hex()}")
print(f"Decryption Key Hex: {decryption_key.hex()}")

if (encryption_key) and (decryption_key):
    plaintext = b'This is a secret message.'

    ciphertext, nonce = encrypt_blowfish_ctr(encryption_key, plaintext)
    print(f"Ciphertext: {ciphertext.hex()}")
    print(f"Nonce: {nonce.hex()}")

    decrypted_text = decrypt_blowfish_ctr(decryption_key, ciphertext, nonce)

    if (decrypted_text == plaintext):
        print("Decryption Successful")
        print(f"Decrypted text: {decrypted_text}")
    else:
        print("Decryption failed")
else:
    print("Key generation failed.")
"""
