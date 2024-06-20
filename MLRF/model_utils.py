import os
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA

from MLRF.features import extract_hog_features, rgb2gray

models = {
    'random_forest': RandomForestClassifier(random_state=42, criterion='entropy', max_depth=10), # criterion, max_depth
    'svm': SVC(max_iter=100 , kernel='rbf', random_state=42), # degree, gamma
    'logistic_regression': LogisticRegression(random_state=42), # penalty, solver
    'knn': KNeighborsClassifier(), # n_neighbors, algorithm
    # 'gradient_boosting': GradientBoostingClassifier(learning_rate=0.01, random_state=42) # loss, max_depth, criterion
}

pipeline = Pipeline([
    # ('contrast', FunctionTransformer(contrast_images, validate=False)),
    # ('grayscale', FunctionTransformer(rgb2gray, validate=False)),
    # ('flatten', ImageFlattener()),
    # ('scaler', StandardScaler()),
    # ('sift_extraction', FunctionTransformer(extract_sift_features, validate=False)),
    ('hog_extraction', FunctionTransformer(extract_hog_features, validate=False)),
    # ('vlad', VLAD(k=16, norming="RN", verbose=False)),
    # ('pca', PCA(n_components=2)),
    ('classifier', None)
])

def save_model(model, model_name, model_dir='models'):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained model to be saved.
        model_name: The name of the model file.
        model_dir: Directory where the model will be saved.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    file_path = os.path.join(model_dir, model_name)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(model_name, model_dir='models'):
    """
    Load a trained model from a file.
    
    Args:
        model_name: The name of the model file.
        model_dir: Directory where the model is saved.
    
    Returns:
        Loaded model.
    """
    file_path = os.path.join(model_dir, model_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No model found at {file_path}")
    
    model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
    return model
