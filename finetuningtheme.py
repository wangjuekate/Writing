import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class FineTunedThemeAnalyzer:
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        """Initialize the theme analyzer with a specific model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.example_clusters = {}
        self.cluster_embeddings = {}
        
    def save_example_clusters(self, filepath: str):
        """Save example clusters to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.example_clusters, f, indent=4)
            
    def load_example_clusters(self, filepath: str):
        """Load example clusters from a JSON file."""
        with open(filepath, 'r') as f:
            self.example_clusters = json.load(f)
        self._update_cluster_embeddings()
    
    def add_example_cluster(self, cluster_name: str, example_texts: List[str]):
        """Add example texts for a specific theme cluster."""
        self.example_clusters[cluster_name] = example_texts
        self._update_cluster_embeddings()
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(text, padding=True, truncation=True, 
                                  return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
                
        return np.array(embeddings)
    
    def _update_cluster_embeddings(self):
        """Update embeddings for all example clusters."""
        for cluster_name, examples in self.example_clusters.items():
            embeddings = self._get_embeddings(examples)
            self.cluster_embeddings[cluster_name] = embeddings.mean(axis=0)
    
    def analyze_themes(self, df: pd.DataFrame, text_column: str, 
                      similarity_threshold: float = 0.7) -> pd.DataFrame:
        """
        Analyze themes in the dataset based on example clusters.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to analyze
            similarity_threshold: Minimum similarity score to assign a theme
            
        Returns:
            DataFrame with theme assignments and similarity scores
        """
        logging.info("Generating embeddings for input texts...")
        text_embeddings = self._get_embeddings(df[text_column].tolist())
        
        # Initialize results
        results = []
        
        for idx, text_embedding in enumerate(text_embeddings):
            similarities = {}
            for theme, cluster_embedding in self.cluster_embeddings.items():
                similarity = cosine_similarity(
                    text_embedding.reshape(1, -1),
                    cluster_embedding.reshape(1, -1)
                )[0][0]
                similarities[theme] = similarity
            
            # Get the most similar theme
            best_theme = max(similarities.items(), key=lambda x: x[1])
            
            results.append({
                'text': df[text_column].iloc[idx],
                'assigned_theme': best_theme[0] if best_theme[1] >= similarity_threshold else 'Other',
                'similarity_score': best_theme[1]
            })
        
        results_df = pd.DataFrame(results)
        return pd.concat([df, results_df[['assigned_theme', 'similarity_score']]], axis=1)
    
    def visualize_theme_distribution(self, df: pd.DataFrame, output_file: str = 'theme_distribution.png'):
        """Create a visualization of theme distribution."""
        theme_counts = df['assigned_theme'].value_counts()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=theme_counts.index, y=theme_counts.values)
        plt.title('Distribution of Themes')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
    def visualize_theme_evolution(self, df: pd.DataFrame, output_file: str = 'theme_evolution.png'):
        """Create a visualization of theme evolution over time."""
        theme_by_year = pd.crosstab(
            df['year'],
            df['assigned_theme'],
            values=df['assigned_theme'],
            aggfunc='count'
        ).fillna(0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(theme_by_year, cmap='YlOrRd', annot=True, fmt='.0f')
        plt.title('Theme Evolution Over Time')
        plt.xlabel('Theme')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

def main():
    # Example usage
    analyzer = FineTunedThemeAnalyzer()
    
    # Add example clusters
    analyzer.add_example_cluster(
        "AI Implementation",
        [
            "Implementation of AI systems in organizational context",
            "Digital transformation and AI adoption",
            "Integration of AI technologies in business processes"
        ]
    )
    
    analyzer.add_example_cluster(
        "Human-AI Interaction",
        [
            "Collaboration between humans and AI systems",
            "User acceptance of AI technologies",
            "Trust in AI-based decision making"
        ]
    )
    
    # Save example clusters for future use
    analyzer.save_example_clusters('example_clusters.json')
    
    # Load and analyze data
    df = pd.read_csv('papers_enhanced.csv')
    
    # Combine title and abstract for analysis
    df['text'] = df['title'] + ' ' + df['abstract'].fillna('')
    
    # Analyze themes
    results_df = analyzer.analyze_themes(df, 'text')
    
    # Save results
    results_df.to_csv('papers_with_fine_tuned_themes.csv', index=False)
    
    # Create visualizations
    analyzer.visualize_theme_distribution(results_df)
    analyzer.visualize_theme_evolution(results_df)

if __name__ == "__main__":
    main()
