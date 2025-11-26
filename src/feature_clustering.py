"""
Feature extraction and clustering for augmented images
Analyzes diversity and quality of augmented datasets
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime


class FeatureExtractor:
    """
    이미지 특징 추출 클래스
    ResNet-based feature extraction for image similarity analysis
    """

    def __init__(self, feature_type: str = 'simple'):
        """
        Args:
            feature_type: 'simple' (histogram + moments) or 'resnet' (deep features)
        """
        self.feature_type = feature_type
        self.model = None

        if feature_type == 'resnet':
            try:
                import torch
                import torchvision
                self.model = torchvision.models.resnet18(pretrained=True)
                # Remove final classification layer
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
                self.model.eval()
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
            except ImportError:
                print("Warning: PyTorch/torchvision not available, falling back to simple features")
                self.feature_type = 'simple'

    def extract_simple_features(self, rgb: np.ndarray) -> np.ndarray:
        """
        간단한 특징 추출: 색상 히스토그램 + 통계적 모멘트

        Args:
            rgb: RGB 이미지 (H, W, 3)

        Returns:
            특징 벡터 (256 dim)
        """
        features = []

        # 1. RGB 히스토그램 (각 채널 32 bins)
        for channel in range(3):
            hist = cv2.calcHist([rgb], [channel], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)
            features.append(hist)

        # 2. HSV 히스토그램 (H: 32, S: 16, V: 16)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])

        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)

        features.extend([hist_h, hist_s, hist_v])

        # 3. 통계적 모멘트 (mean, std, skewness)
        moments = []
        for channel in range(3):
            ch_data = rgb[:, :, channel].flatten()
            moments.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.mean((ch_data - np.mean(ch_data)) ** 3)  # skewness
            ])

        moments = np.array(moments) / 255.0  # Normalize
        features.append(moments)

        # Concatenate all features
        feature_vector = np.concatenate(features)

        return feature_vector

    def extract_resnet_features(self, rgb: np.ndarray) -> np.ndarray:
        """
        ResNet 기반 딥러닝 특징 추출

        Args:
            rgb: RGB 이미지 (H, W, 3)

        Returns:
            특징 벡터 (512 dim)
        """
        import torch
        from torchvision import transforms

        # Preprocessing
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(rgb).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)

        feature_vector = features.squeeze().cpu().numpy()

        return feature_vector

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        """
        특징 추출 (타입에 따라 자동 선택)

        Args:
            rgb: RGB 이미지 (H, W, 3)

        Returns:
            특징 벡터
        """
        if self.feature_type == 'resnet':
            return self.extract_resnet_features(rgb)
        else:
            return self.extract_simple_features(rgb)


class ImageClusterer:
    """
    이미지 클러스터링 클래스
    K-means or DBSCAN clustering for diversity analysis
    """

    def __init__(self, method: str = 'kmeans', n_clusters: int = 5):
        """
        Args:
            method: 'kmeans' or 'dbscan'
            n_clusters: 클러스터 개수 (kmeans only)
        """
        self.method = method
        self.n_clusters = n_clusters
        self.labels = None
        self.cluster_centers = None

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        클러스터링 수행

        Args:
            features: 특징 벡터 배열 (N, feature_dim)

        Returns:
            클러스터 라벨 (N,)
        """
        if self.method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.labels = kmeans.fit_predict(features)
            self.cluster_centers = kmeans.cluster_centers_
        elif self.method == 'dbscan':
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            self.labels = dbscan.fit_predict(features)
            self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self.labels

    def compute_diversity_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        다양성 메트릭 계산

        Args:
            features: 특징 벡터 배열 (N, feature_dim)
            labels: 클러스터 라벨 (N,)

        Returns:
            메트릭 딕셔너리
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        metrics = {}

        # 클러스터 분포
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['n_clusters'] = len(unique_labels)
        metrics['cluster_distribution'] = {int(label): int(count) for label, count in zip(unique_labels, counts)}

        # 클러스터링 품질
        if len(unique_labels) > 1:
            metrics['silhouette_score'] = float(silhouette_score(features, labels))
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(features, labels))
        else:
            metrics['silhouette_score'] = 0.0
            metrics['davies_bouldin_score'] = 0.0

        # Intra-cluster variance (within-cluster spread)
        intra_variances = []
        for label in unique_labels:
            cluster_features = features[labels == label]
            if len(cluster_features) > 1:
                variance = np.var(cluster_features, axis=0).mean()
                intra_variances.append(variance)

        metrics['mean_intra_variance'] = float(np.mean(intra_variances)) if intra_variances else 0.0

        # Inter-cluster distance (between-cluster separation)
        if self.cluster_centers is not None and len(self.cluster_centers) > 1:
            from scipy.spatial.distance import pdist
            inter_distances = pdist(self.cluster_centers)
            metrics['mean_inter_distance'] = float(np.mean(inter_distances))
        else:
            metrics['mean_inter_distance'] = 0.0

        return metrics


def reduce_dimensions(features: np.ndarray, method: str = 'tsne', n_components: int = 2) -> np.ndarray:
    """
    차원 축소 (시각화용)

    Args:
        features: 특징 벡터 배열 (N, feature_dim)
        method: 'tsne' or 'umap'
        n_components: 축소할 차원 (보통 2)

    Returns:
        축소된 특징 (N, n_components)
    """
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        except ImportError:
            print("Warning: UMAP not available, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced = reducer.fit_transform(features)
    return reduced


def analyze_augmentation_quality(
    image_paths: List[Path],
    output_dir: Path,
    feature_type: str = 'simple',
    cluster_method: str = 'kmeans',
    n_clusters: int = 5,
    vis_method: str = 'tsne'
) -> Dict:
    """
    증강된 이미지 품질 분석 (통합 함수)

    Args:
        image_paths: 이미지 경로 리스트
        output_dir: 결과 저장 디렉토리
        feature_type: 특징 추출 방식
        cluster_method: 클러스터링 방식
        n_clusters: 클러스터 개수
        vis_method: 차원 축소 방식

    Returns:
        분석 결과 딕셔너리
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature extraction
    print(f"Extracting features from {len(image_paths)} images...")
    extractor = FeatureExtractor(feature_type=feature_type)
    features_list = []
    valid_paths = []

    for img_path in image_paths:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feature = extractor.extract(img_rgb)
            features_list.append(feature)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Warning: Failed to extract features from {img_path}: {e}")
            continue

    if len(features_list) == 0:
        raise ValueError("No valid images found")

    features = np.array(features_list)
    print(f"Extracted features shape: {features.shape}")

    # 2. Clustering
    print(f"Clustering with {cluster_method}...")
    clusterer = ImageClusterer(method=cluster_method, n_clusters=n_clusters)
    labels = clusterer.fit(features)

    # 3. Diversity metrics
    print("Computing diversity metrics...")
    metrics = clusterer.compute_diversity_metrics(features, labels)

    # 4. Dimensionality reduction for visualization
    print(f"Reducing dimensions with {vis_method}...")
    reduced_features = reduce_dimensions(features, method=vis_method, n_components=2)

    # 5. Find representative images for each cluster
    representatives = {}
    for cluster_id in set(labels):
        if cluster_id == -1:  # Noise in DBSCAN
            continue

        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_features = features[cluster_indices]

        # Find image closest to cluster center
        if clusterer.cluster_centers is not None:
            center = clusterer.cluster_centers[cluster_id]
        else:
            center = cluster_features.mean(axis=0)

        distances = np.linalg.norm(cluster_features - center, axis=1)
        rep_idx = cluster_indices[np.argmin(distances)]

        representatives[int(cluster_id)] = {
            'image_path': str(valid_paths[rep_idx]),
            'cluster_size': int(np.sum(labels == cluster_id))
        }

    # 6. Prepare results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_images': len(valid_paths),
        'feature_type': feature_type,
        'cluster_method': cluster_method,
        'n_clusters': n_clusters,
        'metrics': metrics,
        'representatives': representatives,
        'visualization': {
            'reduced_features': reduced_features.tolist(),
            'labels': labels.tolist(),
            'image_paths': [str(p) for p in valid_paths]
        }
    }

    # Save results as JSON
    results_path = output_dir / 'clustering_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")

    return results
