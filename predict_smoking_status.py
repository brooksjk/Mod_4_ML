import argparse
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import gensim.downloader as api
    WORD2VEC_AVAILABLE = True
except ImportError:
    WORD2VEC_AVAILABLE = False


class SmokingClassifier:
    
    def __init__(self, model_path):
        
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.embedding_dim = model_data['embedding_dim']
        self.label_map = model_data['label_map']
        self.reverse_label_map = model_data['reverse_label_map']
        self.word_vectors = None
        
        print(f"Model type: {self.model_type}")
        print(f"Labels: {list(self.reverse_label_map.values())}")
    
    def load_embeddings(self):
        if WORD2VEC_AVAILABLE:
            try:
                self.word_vectors = api.load('word2vec-google-news-300')
                print(f"Loaded embeddings with {len(self.word_vectors)} words")
            except:
                print("Warning: Could not load embeddings")
    
    def text_to_embedding(self, text):
        if self.word_vectors is None:
            return self.simple_bow_features(text)
        
        words = text.lower().split()
        vectors = []
        
        for word in words:
            if word in self.word_vectors:
                vectors.append(self.word_vectors[word])
        
        if len(vectors) == 0:
            return np.zeros(self.word_vectors.vector_size)
        
        return np.mean(vectors, axis=0)
    
    def simple_bow_features(self, text):
        smoker_terms = ['smoke', 'smoking', 'smoker', 'tobacco', 'cigarette', 
                       'cigar', 'nicotine', 'pack', 'year']
        non_smoker_terms = ['never', 'no', 'not', 'non', 'quit', 'former', 
                           'denies', 'negative', 'zero']
        
        text_lower = text.lower()
        features = []
        
        smoker_count = sum(1 for term in smoker_terms if term in text_lower)
        features.append(smoker_count)
        
        non_smoker_count = sum(1 for term in non_smoker_terms if term in text_lower)
        features.append(non_smoker_count)
        
        features.append(len(text) / 1000.0)
        features.append(len(text.split()) / 100.0)
        
        return np.array(features)
    
    def predict(self, text):
        features = self.text_to_embedding(text).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        
        # Get confidence if available
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0]
            confidence = proba[prediction]
        else:
            # For SVM, use decision function
            if hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(features)[0]
                confidence = 1 / (1 + np.exp(-decision))  # Sigmoid
            else:
                confidence = 1.0
        
        label = self.reverse_label_map[prediction]
        return label, confidence
    
    def predict_batch(self, texts):
        print(f"Processing {len(texts)} documents...")
        features = np.array([self.text_to_embedding(text) for text in tqdm(texts)])
        predictions = self.model.predict(features)
        
        results = []
        for i, pred in enumerate(predictions):
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features[i:i+1])[0]
                confidence = proba[pred]
            else:
                confidence = 1.0
            
            label = self.reverse_label_map[pred]
            results.append((label, confidence))
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Predict smoking status using trained model'
    )
    parser.add_argument('-m', '--model', required=True, help='Path to trained model (.pkl)')
    parser.add_argument('-i', '--input', required=True, help='Input file or directory')
    parser.add_argument('-o', '--output', help='Output file for predictions (CSV)')
    parser.add_argument('--use-embeddings', action='store_true', help='Use word embeddings')
    
    args = parser.parse_args()
    
    # Load classifier
    classifier = SmokingClassifier(args.model)
    
    if args.use_embeddings:
        classifier.load_embeddings()
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        with open(input_path, 'r') as f:
            text = f.read()
        
        label, confidence = classifier.predict(text)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.4f}")
        
    elif input_path.is_dir():
        # Directory of files
        text_files = list(input_path.glob("*.txt"))
        texts = []
        filenames = []
        
        for txt_file in text_files:
            with open(txt_file, 'r') as f:
                texts.append(f.read())
            filenames.append(txt_file.name)
        
        results = classifier.predict_batch(texts)
        
        # Print results
        print("\nPredictions:")
        print("-" * 60)
        for filename, (label, conf) in zip(filenames, results):
            print(f"{filename:40s} {label:15s} {conf:.4f}")
        
        # Save to CSV
        if args.output:
            import csv
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'prediction', 'confidence'])
                for filename, (label, conf) in zip(filenames, results):
                    writer.writerow([filename, label, conf])
            print(f"\nResults saved to {args.output}")
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == '__main__':
    main()