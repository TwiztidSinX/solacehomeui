"""
Parliament Voting System for SolaceOS

This module implements semantic clustering and majority voting for AI Parliament responses.
It groups similar answers using sentence embeddings and finds the consensus response.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple


class ParliamentVoter:
    """
    Performs semantic clustering and voting on Parliament AI responses.

    Uses sentence-transformers to generate embeddings and groups similar responses
    based on cosine similarity. Returns the majority cluster's best answer.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.75):
        """
        Initialize the Parliament voter with a sentence transformer model.

        Args:
            model_name: Name of the sentence-transformers model to use
            similarity_threshold: Minimum cosine similarity to group responses (default: 0.75)
        """
        print(f"Loading Parliament voting model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        print(f"Parliament voter initialized with threshold: {similarity_threshold}")

    def cluster_responses(self, responses: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Cluster responses based on semantic similarity.

        Args:
            responses: List of response dictionaries with 'response' text field

        Returns:
            List of clusters, where each cluster is a list of response indices
        """
        if len(responses) == 0:
            return []

        if len(responses) == 1:
            return [[0]]

        # Extract response texts
        texts = [r.get('response', '') for r in responses]

        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Perform clustering using similarity threshold
        clusters = []
        assigned = set()

        for i in range(len(responses)):
            if i in assigned:
                continue

            # Start a new cluster with this response
            cluster = [i]
            assigned.add(i)

            # Find all similar responses
            for j in range(i + 1, len(responses)):
                if j not in assigned and similarity_matrix[i][j] >= self.similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)

            clusters.append(cluster)

        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)

        return clusters

    def find_majority(self, clusters: List[List[int]], min_votes: int = 4) -> Tuple[int, List[int]]:
        """
        Find the majority cluster (cluster with most votes).

        Args:
            clusters: List of clusters from cluster_responses()
            min_votes: Minimum number of votes to be considered a majority

        Returns:
            Tuple of (cluster_index, cluster) or (-1, []) if no majority found
        """
        if not clusters:
            return (-1, [])

        # Largest cluster is already first due to sorting
        largest_cluster = clusters[0]

        if len(largest_cluster) >= min_votes:
            return (0, largest_cluster)
        else:
            # No majority, return largest cluster anyway
            return (0, largest_cluster)

    def vote(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform voting on Parliament responses.

        Args:
            responses: List of response dictionaries with fields:
                - model: Name of the AI model
                - response: The response text
                - confidence: Confidence score (0.0 to 1.0)

        Returns:
            Dictionary with voting results:
                - winning_answer: The selected response text
                - winning_model: Name of the winning model
                - confidence: Confidence score of the winning answer
                - votes: Number of models in the winning cluster
                - total_clusters: Total number of distinct clusters
                - cluster_details: List of cluster information
        """
        if not responses:
            return {
                'winning_answer': '',
                'winning_model': '',
                'confidence': 0.0,
                'votes': 0,
                'total_clusters': 0,
                'cluster_details': []
            }

        # Handle single response
        if len(responses) == 1:
            r = responses[0]
            return {
                'winning_answer': r.get('response', ''),
                'winning_model': r.get('model', 'unknown'),
                'confidence': r.get('confidence', 0.5),
                'votes': 1,
                'total_clusters': 1,
                'cluster_details': [{
                    'models': [r.get('model', 'unknown')],
                    'size': 1,
                    'avg_confidence': r.get('confidence', 0.5)
                }]
            }

        # Cluster responses
        clusters = self.cluster_responses(responses)

        # Find majority cluster
        majority_idx, majority_cluster = self.find_majority(clusters, min_votes=4)

        # Build cluster details
        cluster_details = []
        for cluster_indices in clusters:
            cluster_models = [responses[i].get('model', 'unknown') for i in cluster_indices]
            cluster_confidences = [responses[i].get('confidence', 0.5) for i in cluster_indices]
            avg_confidence = np.mean(cluster_confidences) if cluster_confidences else 0.5

            cluster_details.append({
                'models': cluster_models,
                'size': len(cluster_indices),
                'avg_confidence': float(avg_confidence)
            })

        # Select best response from winning cluster (highest confidence)
        winning_cluster = majority_cluster if majority_cluster else clusters[0]

        # Find all responses with max confidence to handle ties
        max_confidence = max((responses[idx].get('confidence', 0.5) for idx in winning_cluster), default=0.5)
        tied_responses = [idx for idx in winning_cluster if responses[idx].get('confidence', 0.5) == max_confidence]

        # If multiple responses have same max confidence, randomly pick one for fairness
        import random
        best_response_idx = random.choice(tied_responses) if tied_responses else winning_cluster[0]
        best_confidence = responses[best_response_idx].get('confidence', 0.5)

        winning_response = responses[best_response_idx]

        return {
            'winning_answer': winning_response.get('response', ''),
            'winning_model': winning_response.get('model', 'unknown'),
            'confidence': best_confidence,
            'votes': len(winning_cluster),
            'total_clusters': len(clusters),
            'cluster_details': cluster_details
        }


def demo():
    """Demo function to test the voting system."""
    # Mock responses for testing
    mock_responses = [
        {"model": "gpt-4", "response": "Python is better because it's readable and has great libraries", "confidence": 0.95},
        {"model": "claude-3", "response": "Python is superior due to readability and extensive ecosystem", "confidence": 0.92},
        {"model": "gemini", "response": "I prefer Python for its clean syntax and community support", "confidence": 0.90},
        {"model": "qwen", "response": "Python wins with readable code and powerful libraries", "confidence": 0.88},
        {"model": "deepseek", "response": "JavaScript is faster for web applications and has better performance", "confidence": 0.85},
        {"model": "grok", "response": "JS performs better in browsers and has async capabilities", "confidence": 0.87},
    ]

    voter = ParliamentVoter()
    result = voter.vote(mock_responses)

    print("\n=== Parliament Voting Results ===")
    print(f"Winning Answer: {result['winning_answer'][:80]}...")
    print(f"Winning Model: {result['winning_model']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Votes: {result['votes']}/{len(mock_responses)}")
    print(f"Total Clusters: {result['total_clusters']}")
    print("\nCluster Details:")
    for i, cluster in enumerate(result['cluster_details']):
        print(f"  Cluster {i+1}: {cluster['size']} votes, avg confidence: {cluster['avg_confidence']:.2f}")
        print(f"    Models: {', '.join(cluster['models'])}")


if __name__ == "__main__":
    demo()
