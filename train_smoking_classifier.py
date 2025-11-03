import argparse
import os
import json
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    import gensim.downloader as api
    WORD2VEC_AVAILABLE = True
except ImportError:
    WORD2VEC_AVAILABLE = False
    print("Warning: gensim not available. Install with: pip install gensim")


class SmokingClassifierTrainer:
    def __init__(self, embedding_dim=100, model_type='logistic'):
        
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        self.model = None
        self.word_vectors = None
        self.label_map = {'NON_SMOKER': 0, 'SMOKER': 1}
        self.reverse_label_map = {0: 'NON_SMOKER', 1: 'SMOKER'}
        
    def load_embeddings(self):
        print("Loading word embeddings...")
        if WORD2VEC_AVAILABLE:
            try:
                self.word_vectors = api.load('word2vec-google-news-300')
                print(f"Loaded word2vec with {len(self.word_vectors)} words")
            except:
                print("Could not load embeddings. Using random initialization.")
                self.word_vectors = None
        else:
            print("Using TF-IDF features instead of embeddings")
            self.word_vectors = None
    
    def text_to_embedding(self, text):
        # Following Eq. 3 from Wang et al.: x = (1/M) * sum(x_i)
    
        if self.word_vectors is None:
            # Fallback to simple bag-of-words representation
            return self.simple_bow_features(text)
        
        words = text.lower().split()
        vectors = []
        
        for word in words:
            if word in self.word_vectors:
                vectors.append(self.word_vectors[word])
        
        if len(vectors) == 0:
            # Return zero vector if no words found
            return np.zeros(self.word_vectors.vector_size)
        
        # Mean pooling as per Wang et al.
        return np.mean(vectors, axis=0)
    
    def simple_bow_features(self, text):
        # Key smoking-related terms
        smoker_terms = ['smoke', 'smoking', 'smoker', 'tobacco', 'cigarette', 
                       'cigar', 'nicotine', 'pack', 'year']
        non_smoker_terms = ['never', 'no', 'not', 'non', 'quit', 'former', 
                           'denies', 'negative', 'zero']
        
        text_lower = text.lower()
        features = []
        
        # Count smoker terms
        smoker_count = sum(1 for term in smoker_terms if term in text_lower)
        features.append(smoker_count)
        
        # Count non-smoker terms
        non_smoker_count = sum(1 for term in non_smoker_terms if term in text_lower)
        features.append(non_smoker_count)
        
        # Character length (normalized)
        features.append(len(text) / 1000.0)
        
        # Word count
        features.append(len(text.split()) / 100.0)
        
        return np.array(features)
    
    def load_weak_labels(self, xmi_dir):
        
        # parse XMI to extract labels
        from xml.etree import ElementTree as ET
        
        texts = []
        labels = []
        
        xmi_files = list(Path(xmi_dir).glob("*.xmi"))
        
        for xmi_file in tqdm(xmi_files):
            txt_file = xmi_file.with_suffix('.txt')
            if not txt_file.exists():
                continue
                
            with open(txt_file, 'r') as f:
                text = f.read()
            
            # Parse XMI to get label
            tree = ET.parse(xmi_file)
            root = tree.getroot()
            
            # Find EntityMention annotations
            label = None
            for entity in root.findall(".//{*}EntityMention"):
                entity_type = entity.get('entityType')
                if entity_type in ['SMOKER', 'NON_SMOKER']:
                    label = entity_type
                    break
            
            # Default to NON_SMOKER if no annotation found
            if label is None:
                label = 'NON_SMOKER'
            
            texts.append(text)
            labels.append(self.label_map[label])
        
        print(f"Loaded {len(texts)} documents")
        return texts, labels
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
       
        print(f"\nTraining {self.model_type} model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        
        # Initialize model based on type
        if self.model_type == 'logistic':
            # Following Wang et al., using L2 regularization
            self.model = LogisticRegression(
                C=10.0,  # Regularization parameter from paper
                max_iter=1000,
                random_state=123
            )
        elif self.model_type == 'random_forest':
            # Random Forest with 5 trees as per paper
            self.model = RandomForestClassifier(
                n_estimators=5,
                random_state=123
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate on training data
        train_pred = self.model.predict(X_train)
        train_acc = (train_pred == y_train).mean()
        print(f"Training accuracy: {train_acc:.4f}")
        
        # Evaluate on validation
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = (val_pred == y_val).mean()
            print(f"Validation accuracy: {val_acc:.4f}")
            print("\nValidation Classification Report:")
            print(classification_report(y_val, val_pred, target_names=['NON_SMOKER', 'SMOKER']))
    
    def save_model(self, output_path):
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'embedding_dim': self.embedding_dim,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train smoking classifier with weak supervision'
    )
    parser.add_argument('-i', '--input-dir', required=True, help='Directory containing XMI files with weak labels')
    parser.add_argument('-t', '--text-dir', required=True, help='Directory containing original text files')
    parser.add_argument('-o', '--output-model', required=True, help='Output path for trained model (.pkl)')
    parser.add_argument('-m', '--model-type', choices=['logistic', 'svm', 'random_forest'], default='logistic', help='Type of model to train')
    parser.add_argument('--test-split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--use-embeddings', action='store_true', help='Use word embeddings (requires gensim)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SmokingClassifierTrainer(model_type=args.model_type)
    
    # Load embeddings if requested
    if args.use_embeddings:
        trainer.load_embeddings()
    
    # Load weakly labeled data
    texts, labels = trainer.load_weak_labels(args.input_dir)
    
    # Convert texts to features
    print("\nExtracting features...")
    X = np.array([trainer.text_to_embedding(text) for text in tqdm(texts)])
    y = np.array(labels)
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )
    
    print(f"\nClass distribution in training:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = trainer.reverse_label_map[label]
        print(f"  {label_name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Train model
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Save model
    trainer.save_model(args.output_model)


if __name__ == '__main__':
    main()

# python3 train_smoking_classifier.py -i /scratch/jkbrook/NLP/Mod_4_ML/output -t /scratch/jkbrook/NLP/Mod_4_ML/output -o model/V2_smoking_classifier_RF_embed.pkl -m random_forest --use-embeddings
