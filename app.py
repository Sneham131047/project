from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# Load a reference image (genuine note)
reference_image_path = "image223.jpeg"
reference_image = cv2.imread(reference_image_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:
        # Convert uploaded image to numpy array
        image = np.frombuffer(file.read(), np.uint8)
        user_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Process both the user image and reference image to compare
        similarity_percentage = compare_images(user_image, reference_image)

        return jsonify({'similarity': f"{similarity_percentage:.2f}%"})
    return jsonify({'error': 'No file uploaded'})

def compare_images(user_image, reference_image):
    """
    Compares two images (user uploaded image and reference image) and returns a similarity percentage.
    """
    
    # 1. Convert both images to grayscale
    gray_user = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize the images to be the same size (optional, if sizes differ)
    gray_user_resized = cv2.resize(gray_user, (gray_ref.shape[1], gray_ref.shape[0]))

    # 3. Calculate Structural Similarity Index (SSIM) between two images
    (score, diff) = ssim(gray_user_resized, gray_ref, full=True)
    similarity_score = score * 100  # SSIM returns a score between -1 and 1, convert to percentage

    # 4. Feature Matching using ORB (Optional: Add keypoint-based feature matching)
    orb_similarity = orb_feature_matching(user_image, reference_image)

    # Average similarity from both methods
    overall_similarity = (similarity_score + orb_similarity) / 2

    return overall_similarity

def orb_feature_matching(user_image, reference_image):
    """
    Compares two images using ORB (keypoints and descriptors) and returns a similarity score.
    """
    
    # 1. Initialize ORB detector
    orb = cv2.ORB_create()

    # 2. Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(user_image, None)
    kp2, des2 = orb.detectAndCompute(reference_image, None)

    # 3. Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 4. Match descriptors
    matches = bf.match(des1, des2)

    # 5. Sort them in the order of their distance (the lower the distance, the better the match)
    matches = sorted(matches, key=lambda x: x.distance)

    # 6. Calculate similarity based on matches
    num_matches = len(matches)
    total_keypoints = max(len(kp1), len(kp2))
    
    # Return the match percentage as a ratio of keypoints matched
    orb_similarity_percentage = (num_matches / total_keypoints) * 100 if total_keypoints > 0 else 0
    
    return orb_similarity_percentage

if __name__ == '_main_':
    app.run(debug=True)
