
"""
Theme Analysis using BERT and clustering
"""
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import umap
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_bert_embeddings(texts, model_name='allenai/scibert_scivocab_uncased'):
    """Get BERT embeddings for a list of texts"""
    logging.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    embeddings = []
    batch_size = 8
    
    logging.info("Generating embeddings...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            # Use CLS token embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def find_optimal_clusters(embeddings, max_clusters=15):
    """Find optimal number of clusters using silhouette score"""
    logging.info("Finding optimal number of clusters...")
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters

def extract_cluster_keywords(texts, labels, top_n=10):
    """Extract keywords characterizing each cluster using TF-IDF"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    cluster_keywords = {}
    for label in set(labels):
        cluster_docs = tfidf_matrix[labels == label]
        tfidf_mean = cluster_docs.mean(axis=0).A1
        top_indices = tfidf_mean.argsort()[-top_n:][::-1]
        cluster_keywords[label] = [feature_names[i] for i in top_indices]
    
    return cluster_keywords

def visualize_clusters(embeddings, labels, titles, save_path='cluster_visualization.png'):
    """Visualize clusters using UMAP"""
    logging.info("Creating cluster visualization...")
    
    # Reduce dimensionality for visualization
    reducer = umap.UMAP(random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab20')
    
    # Add title tooltips
    for i, title in enumerate(titles):
        plt.annotate(title[:50] + '...', (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.colorbar(scatter)
    plt.title('Cluster Visualization of Research Themes')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_themes(csv_file):
    """Main function to analyze themes in the papers"""
    # Load data
    logging.info(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Combine title and abstract for better theme detection
    texts = []
    for _, row in df.iterrows():
        text = f"{row['title']} {row['abstract']}"
        texts.append(text.strip())
    
    # Get BERT embeddings
    embeddings = get_bert_embeddings(texts)
    
    # Find optimal number of clusters
    n_clusters = find_optimal_clusters(embeddings)
    logging.info(f"Optimal number of clusters: {n_clusters}")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # Extract keywords for each cluster
    cluster_keywords = extract_cluster_keywords(texts, labels)
    
    # Create visualization
    visualize_clusters(embeddings, labels, df['title'])
    
    # Add cluster labels to DataFrame
    df['cluster'] = labels
    
    # Print cluster summaries
    print("\nTheme Analysis Results:")
    print("-" * 50)
    for cluster in range(n_clusters):
        cluster_papers = df[df['cluster'] == cluster]
        print(f"\nTheme {cluster + 1}:")
        print("Keywords:", ", ".join(cluster_keywords[cluster]))
        print(f"Number of papers: {len(cluster_papers)}")
        print("\nTop cited papers in this theme:")
        top_papers = cluster_papers.nlargest(3, 'times_cited')[['title', 'times_cited', 'year']]
        print(top_papers.to_string(index=False))
        print("-" * 50)
    
    # Save results
    df.to_csv('papers_with_themes.csv', index=False)
    
    # Create theme evolution plot
    plt.figure(figsize=(12, 6))
    theme_by_year = df.pivot_table(
        index='year', 
        columns='cluster', 
        values='title',
        aggfunc='count'
    ).fillna(0)
    
    sns.heatmap(theme_by_year, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Theme Evolution Over Time')
    plt.xlabel('Theme')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.savefig('theme_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Install required packages if not already installed
    import subprocess
    import sys
    
    required_packages = [
        'transformers',
        'torch',
        'scikit-learn',
        'umap-learn',
        'seaborn'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    csv_file = "papers_enhanced.csv"
    analyze_themes(csv_file)
