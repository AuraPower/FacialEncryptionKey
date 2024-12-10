import cv2 as cv
import dlib
import numpy as np
from scipy.spatial import distance

def extract_face_embedding(image_path, model_path="..\\Models\\dlib_face_recognition_resnet_model_v1.dat"):
    """
    Extract face embeddings from an image using dlib's pre-trained model.
    
    Args:
        image_path (str): Path to the image.
        model_path (str): Path to dlib's pre-trained face recognition model.
    
    Returns:
        tuple: A list of face embeddings and the corresponding bounding boxes.
    """
    # Load the dlib face detector and face embedding model
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("..\\Models\\shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1(model_path)

    # Load and preprocess the image
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open or read the image: {image_path}")
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Detect faces
    faces = detector(img_rgb)
    if len(faces) == 0:
        print(f"No faces detected in {image_path}.")
        return [], []

    embeddings = []
    boxes = []

    face = faces[0]

    # Get facial landmarks
    shape = shape_predictor(img_rgb, face)
    # Compute the embedding
    embedding = face_rec_model.compute_face_descriptor(img_rgb, shape)
    embeddings.append(np.array(embedding))
    boxes.append((face.left(), face.top(), face.right(), face.bottom()))

    return embeddings, boxes


def compare_faces(embedding1, embedding2, threshold=0.1):
    """
    Compare two face embeddings to check if they represent the same person.
    
    Args:
        embedding1 (numpy array): First face embedding.
        embedding2 (numpy array): Second face embedding.
        threshold (float): Distance threshold for similarity.
    
    Returns:
        bool: True if the faces match, False otherwise.
    """
    dist = distance.euclidean(embedding1, embedding2)
    return dist < threshold, dist


def run_comparison(image1_path, image2_path):
    # Load embeddings for both images
    embeddings1, boxes1 = extract_face_embedding("..\\headshot\\"+image1_path)
    embeddings2, boxes2 = extract_face_embedding("..\\headshot\\"+image2_path)

    if not embeddings1 or not embeddings2:
        print("Could not extract embeddings from one or both images.")
        return

    # Compare embeddings
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            is_same, dist = compare_faces(emb1, emb2)
            print(f"Face {i+1} in Image 1 and Face {j+1} in Image 2 - Same Person: {is_same}, Distance: {dist}")
            return True


# Example Usage
resolutions = ['500x500', '1000x1000', '2000x2000']
extentions = ['.png', '.tif', '.bmp', '.jpg']

enc_path = resolutions[2]+extentions[3]
dec_path = resolutions[2]+extentions[2]
# run_comparison(enc_path, dec_path)
