# Module 4 Machine Learning Exercises: How to Run
Below are instructions to reproduce or deploy the model on either macOS or Linux (RedHat/headless):

Requirements:
- Python 3.8+
- Conda or venv recommended
- Required packages:
`pip install scikit-learn pandas numpy tqdm medspacy spacy gensim`

Saved model = `V2_smoking_classifier_RF_embed.pkl`

Steps:
1. unzip the project folder

2. Downloaded pretrained embeddings if using the `--use-embeddings` flag
```
python -m gensim.downloader --download glove-wiki-gigaword-100
```

3. Run the prepreprocessing scripts:
```
# Generate XMIs
python3 medspaCy_regex.py --input notes/ --output output/

# Get .txt files from discharge.csv
python3 extract_patient_notes_from_csv.py --input discharge.csv --output output/
```

4. Train/Evaluate the Model:
```
python3 train_smoking_classifier.py -i output/ -t output/ \
-o model/V2_smoking_classifier_RF_embed.pkl \
-m random_forest --use-embeddings
```

5. Use Classifier:
```
python3 predict_smoking_status.py \
--model models/smoking_classifier.pkl \
--input notes_to_classify/note_001.txt
```
