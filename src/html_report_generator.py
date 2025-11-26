"""
HTML report generator for augmentation quality analysis
Creates interactive reports with visualizations
"""

import json
from pathlib import Path
from typing import Dict, List
import base64
import numpy as np


def encode_image_to_base64(image_path: Path, max_size: int = 400) -> str:
    """
    이미지를 base64로 인코딩 (HTML 임베딩용)

    Args:
        image_path: 이미지 경로
        max_size: 최대 크기 (리사이즈)

    Returns:
        base64 인코딩된 문자열
    """
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return ""

    # Resize for faster loading
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

    # Encode to base64
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return f"data:image/jpeg;base64,{img_base64}"


def generate_scatter_plot_html(
    reduced_features: np.ndarray,
    labels: np.ndarray,
    image_paths: List[str]
) -> str:
    """
    t-SNE/UMAP 산점도 HTML 생성 (Plotly 사용)

    Args:
        reduced_features: 축소된 특징 (N, 2)
        labels: 클러스터 라벨 (N,)
        image_paths: 이미지 경로 리스트

    Returns:
        HTML 문자열
    """
    # Prepare data for Plotly
    x_coords = reduced_features[:, 0].tolist()
    y_coords = reduced_features[:, 1].tolist()
    labels_list = labels.tolist()

    # Unique cluster labels
    unique_labels = sorted(set(labels_list))

    # Generate color palette
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # Build traces for each cluster
    traces = []
    for idx, label in enumerate(unique_labels):
        cluster_indices = [i for i, l in enumerate(labels_list) if l == label]
        cluster_x = [x_coords[i] for i in cluster_indices]
        cluster_y = [y_coords[i] for i in cluster_indices]
        cluster_paths = [image_paths[i] for i in cluster_indices]

        trace = {
            'x': cluster_x,
            'y': cluster_y,
            'mode': 'markers',
            'marker': {
                'size': 10,
                'color': colors[idx % len(colors)],
                'line': {'width': 1, 'color': 'white'}
            },
            'name': f'Cluster {label}',
            'text': [Path(p).name for p in cluster_paths],
            'hovertemplate': '<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        }
        traces.append(trace)

    # Convert to JSON
    traces_json = json.dumps(traces)

    html = f"""
    <div id="scatter-plot" style="width:100%;height:600px;"></div>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <script>
        var traces = {traces_json};
        var layout = {{
            title: 'Feature Space Visualization (2D Projection)',
            xaxis: {{ title: 'Dimension 1' }},
            yaxis: {{ title: 'Dimension 2' }},
            hovermode: 'closest',
            showlegend: true,
            legend: {{ x: 1.02, y: 1 }}
        }};
        Plotly.newPlot('scatter-plot', traces, layout);
    </script>
    """

    return html


def generate_html_report(
    results: Dict,
    output_path: Path,
    include_images: bool = True,
    max_images_per_cluster: int = 5
) -> None:
    """
    HTML 품질 보고서 생성

    Args:
        results: clustering_results.json 내용
        output_path: 저장 경로 (HTML 파일)
        include_images: 이미지 임베딩 여부
        max_images_per_cluster: 클러스터당 표시할 최대 이미지 수
    """
    # Extract data
    metrics = results['metrics']
    representatives = results['representatives']
    vis_data = results['visualization']

    reduced_features = np.array(vis_data['reduced_features'])
    labels = np.array(vis_data['labels'])
    image_paths = vis_data['image_paths']

    # Generate scatter plot
    scatter_html = generate_scatter_plot_html(reduced_features, labels, image_paths)

    # HTML template
    html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Augmentation Quality Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            color: white;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .cluster-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .cluster-card {{
            background: #fff;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .cluster-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .cluster-header {{
            background: #667eea;
            color: white;
            padding: 15px;
            font-weight: bold;
        }}
        .cluster-images {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            padding: 15px;
        }}
        .cluster-image {{
            width: 100%;
            height: 120px;
            object-fit: cover;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .no-image {{
            width: 100%;
            height: 120px;
            background: #f0f0f0;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            color: #999;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        .good {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .bad {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Data Augmentation Quality Report</h1>
        <p>Generated: {results['timestamp']}</p>
        <p>Total Images: {results['n_images']} | Clusters: {metrics['n_clusters']} | Method: {results['cluster_method'].upper()}</p>
    </div>

    <div class="section">
        <h2>📊 Diversity Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Number of Clusters</h3>
                <div class="value">{metrics['n_clusters']}</div>
            </div>
            <div class="metric-card">
                <h3>Silhouette Score</h3>
                <div class="value {'good' if metrics.get('silhouette_score', 0) > 0.5 else 'warning' if metrics.get('silhouette_score', 0) > 0.3 else 'bad'}">{metrics.get('silhouette_score', 0):.3f}</div>
                <small>Range: [-1, 1], Higher is better</small>
            </div>
            <div class="metric-card">
                <h3>Davies-Bouldin Score</h3>
                <div class="value {'good' if metrics.get('davies_bouldin_score', 10) < 1.0 else 'warning' if metrics.get('davies_bouldin_score', 10) < 2.0 else 'bad'}">{metrics.get('davies_bouldin_score', 0):.3f}</div>
                <small>Lower is better</small>
            </div>
            <div class="metric-card">
                <h3>Mean Intra-Variance</h3>
                <div class="value">{metrics.get('mean_intra_variance', 0):.3f}</div>
                <small>Within-cluster spread</small>
            </div>
        </div>

        <h3 style="margin-top: 30px;">Cluster Size Distribution</h3>
        <table>
            <thead>
                <tr>
                    <th>Cluster ID</th>
                    <th>Number of Images</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add cluster distribution rows
    total_images = results['n_images']
    for cluster_id, count in metrics['cluster_distribution'].items():
        percentage = (count / total_images) * 100
        html_content += f"""
                <tr>
                    <td>Cluster {cluster_id}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>🗺️ Feature Space Visualization</h2>
        <p>2D projection of high-dimensional features using t-SNE/UMAP. Each point represents an image, colored by cluster.</p>
"""
    html_content += scatter_html
    html_content += """
    </div>

    <div class="section">
        <h2>🖼️ Representative Images by Cluster</h2>
        <p>Sample images closest to each cluster center</p>
        <div class="cluster-grid">
"""

    # Add cluster representative images
    for cluster_id, rep_data in representatives.items():
        cluster_size = rep_data['cluster_size']
        rep_image_path = Path(rep_data['image_path'])

        # Find all images in this cluster
        cluster_indices = [i for i, l in enumerate(labels) if l == int(cluster_id)]
        cluster_image_paths = [Path(image_paths[i]) for i in cluster_indices[:max_images_per_cluster]]

        html_content += f"""
            <div class="cluster-card">
                <div class="cluster-header">
                    Cluster {cluster_id} ({cluster_size} images)
                </div>
                <div class="cluster-images">
"""

        if include_images:
            for img_path in cluster_image_paths:
                try:
                    img_base64 = encode_image_to_base64(img_path)
                    if img_base64:
                        html_content += f"""
                    <img src="{img_base64}" class="cluster-image" alt="{img_path.name}" title="{img_path.name}">
"""
                    else:
                        html_content += f"""
                    <div class="no-image">{img_path.name}</div>
"""
                except Exception as e:
                    html_content += f"""
                    <div class="no-image">Error loading image</div>
"""
        else:
            for img_path in cluster_image_paths:
                html_content += f"""
                    <div class="no-image">{img_path.name}</div>
"""

        html_content += """
                </div>
            </div>
"""

    html_content += """
        </div>
    </div>

    <div class="section">
        <h2>📝 Analysis Summary</h2>
        <h3>Quality Indicators:</h3>
        <ul>
"""

    # Add quality interpretations
    silhouette = metrics.get('silhouette_score', 0)
    if silhouette > 0.5:
        html_content += """
            <li class="good">✅ <strong>Good cluster separation</strong> - Augmented images show distinct variations</li>
"""
    elif silhouette > 0.3:
        html_content += """
            <li class="warning">⚠️ <strong>Moderate cluster separation</strong> - Consider more diverse augmentation</li>
"""
    else:
        html_content += """
            <li class="bad">❌ <strong>Poor cluster separation</strong> - Augmentation may be too similar</li>
"""

    db_score = metrics.get('davies_bouldin_score', 10)
    if db_score < 1.0:
        html_content += """
            <li class="good">✅ <strong>Well-separated clusters</strong> - Good diversity achieved</li>
"""
    elif db_score < 2.0:
        html_content += """
            <li class="warning">⚠️ <strong>Some cluster overlap</strong> - Acceptable but could improve</li>
"""
    else:
        html_content += """
            <li class="bad">❌ <strong>Significant cluster overlap</strong> - Consider different augmentation strategies</li>
"""

    # Check balance
    cluster_sizes = list(metrics['cluster_distribution'].values())
    min_size = min(cluster_sizes)
    max_size = max(cluster_sizes)
    balance_ratio = min_size / max_size if max_size > 0 else 0

    if balance_ratio > 0.5:
        html_content += """
            <li class="good">✅ <strong>Balanced cluster distribution</strong> - Augmentation covers variations evenly</li>
"""
    else:
        html_content += """
            <li class="warning">⚠️ <strong>Imbalanced clusters</strong> - Some variations are over-represented</li>
"""

    html_content += """
        </ul>

        <h3>Recommendations:</h3>
        <ul>
            <li>For training: Aim for Silhouette > 0.4 and balanced cluster sizes</li>
            <li>If clusters overlap too much: Increase augmentation strength or add more variation types</li>
            <li>If one cluster dominates: Adjust augmentation parameters to create more diversity</li>
            <li>Monitor this report across multiple augmentation runs to optimize settings</li>
        </ul>
    </div>

    <div class="footer">
        Generated by SAM 3D GUI - Data Augmentation Quality Analyzer
    </div>
</body>
</html>
"""

    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML report generated: {output_path}")
