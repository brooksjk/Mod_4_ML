import glob
import os
import argparse
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter

class DataBuilder(object):
    def __init__(self, corpus, test_split=0.2, seed=123):
        self.corpus = corpus
        self.test_split = test_split
        self.seed = seed

    @staticmethod
    def read_note(file_dir):
        with open(file_dir, 'r') as f:
            return f.read()

    def read_dir(self, label_dir):
        examples = []
        full_dir = os.path.join(self.corpus, label_dir, "*.txt")
        for file_path in glob.glob(full_dir):
            text = self.read_note(file_path)
            examples.append(text)
        return examples

    def get_data(self):
        # Load each class
        examples_breast = self.read_dir('breastca')
        examples_pancreas = self.read_dir('pdac')

        all_examples = examples_breast + examples_pancreas
        all_labels = [0]*len(examples_breast) + [1]*len(examples_pancreas)

        # Stratified split
        examples_train, examples_test, labels_train, labels_test = train_test_split(
            all_examples,
            all_labels,
            test_size=self.test_split,
            random_state=self.seed,
            stratify=all_labels 
        )

        print("Class counts:")
        print(f"  Total: {Counter(all_labels)}")
        print(f"  Train set: {Counter(labels_train)}")
        print(f"  Test set: {Counter(labels_test)}")

        return list(examples_train), list(labels_train), list(examples_test), list(labels_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build bag-of-words dataset for CORAL (stratified split)")
    parser.add_argument("--input_dir", required=True, help="Path to corpus directory (with breastca/ and pdac/)")
    parser.add_argument("--output_file", required=True, help="Where to save the output pickle file")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction of data to reserve for testing")
    args = parser.parse_args()

    builder = DataBuilder(corpus=args.input_dir, test_split=args.test_split)
    examples_train, labels_train, examples_test, labels_test = builder.get_data()

    with open(args.output_file, "wb") as f:
        pickle.dump({
            "examples_train": examples_train,
            "labels_train": labels_train,
            "examples_test": examples_test,
            "labels_test": labels_test
        }, f)

    print(f"\nSaved dataset to {args.output_file}")
    print(f"Training examples: {len(examples_train)} | Test examples: {len(examples_test)}")
