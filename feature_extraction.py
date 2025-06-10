import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from scipy import signal as sg
from tqdm import tqdm

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return image, gray

def extract_lbp(gray):
    lbp = local_binary_pattern(gray, P=8, R=1)
    hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 256))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def extract_glcm(gray):
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=False, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]
    return np.array(features)

def extract_hog(image):
    resized = cv2.resize(image, (128, 128))
    features = hog(resized,
                   orientations=8,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1),
                   visualize=False,
                   channel_axis=-1)
    return features

def extract_laws(gray):
    (rows, cols) = gray.shape[:2]
    smooth_kernel = (1/25)*np.ones((5,5))
    gray_smooth = sg.convolve(gray, smooth_kernel,"same")
    gray_processed = np.abs(gray - gray_smooth)
    
    filter_vectors = np.array([[ 1,  4,  6,  4, 1],
                               [-1, -2,  0,  2, 1],
                               [-1,  0,  2,  0, 1],
                               [ 1, -4,  6, -4, 1]])

    filters = [np.matmul(fv1.reshape(5, 1), fv2.reshape(1, 5))
               for fv1 in filter_vectors for fv2 in filter_vectors]

    conv_maps = np.zeros((rows, cols, 16))
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], 'same')

    texture_maps = [
        (conv_maps[:, :, 1]+conv_maps[:, :, 4])//2,
        (conv_maps[:, :, 2]+conv_maps[:, :, 8])//2,
        (conv_maps[:, :, 3]+conv_maps[:, :, 12])//2,
        (conv_maps[:, :, 7]+conv_maps[:, :, 13])//2,
        (conv_maps[:, :, 6]+conv_maps[:, :, 9])//2,
        (conv_maps[:, :, 11]+conv_maps[:, :, 14])//2,
        conv_maps[:, :, 10],
        conv_maps[:, :, 5],
        conv_maps[:, :, 15]
    ]
    norm_map = conv_maps[:, :, 0]
    TEM = [np.abs(tm).sum() / np.abs(norm_map).sum() for tm in texture_maps]
    TEM = np.array(TEM)
    TEM = TEM / np.linalg.norm(TEM)
    return TEM

def save_features(method, features, labels):
    os.makedirs("./saved_features", exist_ok=True)
    save_path = f"./saved_features/{method}_features.npz"
    np.savez_compressed(save_path, features=features, labels=labels)
    print(f"Saved {method} features to {save_path}")

def process_dataset(dataset_path, method):
    features = []
    labels = []
    class_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    for cls in class_dirs:
        class_path = os.path.join(dataset_path, cls)
        image_files = sorted([f for f in os.listdir(class_path) if f.endswith('.png')])
        for img_file in tqdm(image_files, desc=f"{cls:12s} - {method.upper()}"):
            img_path = os.path.join(class_path, img_file)
            image, gray = preprocess_image(img_path)

            if method == 'lbp':
                feature = extract_lbp(gray)
            elif method == 'glcm':
                feature = extract_glcm(gray)
            elif method == 'hog':
                feature = extract_hog(image)
            elif method == 'laws':
                feature = extract_laws(gray)
            else:
                continue

            features.append(feature)
            labels.append(cls)

    save_features(method, np.array(features), np.array(labels))

if __name__ == "__main__":
    dataset_path = "./recaptcha-dataset/Large"
    methods = ["lbp", "glcm", "hog", "laws"]
    for method in methods:
        process_dataset(dataset_path, method)