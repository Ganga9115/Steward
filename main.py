import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer
import spacy
import json


class TransactionModelSDK:
    def __init__(self, model_path='model.pkl', data_path='transactions.csv', extra_data_path = 'extra_transactions.csv', feedback_path = 'feedback_storage.csv', feedback_threshold=100):

        """
        Initialize the SDK.
        :param model_path: Where the trained model is saved.
        :param data_path: Where the dataset is stored.
        :param feedback_threshold: How many feedback items to wait for before retraining.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.extra_data_path = extra_data_path
        self.feedback_path = feedback_path
        self.feedback_threshold = feedback_threshold
        
        self.model = None
        self.label_encoder = None

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading Spacy model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Load the model immediately if it exists
        self.load_model()
    # --- FUNCTION 1: LOAD MODEL ---

    def _preprocess(self, text):
        """
        Private helper to clean text using Spacy.
        Removes stop words, punctuation, and performs lemmatization.
        """
        # Convert to string to handle potential NaNs or numbers
        text = str(text).lower() 
        doc = self.nlp(text)
        
        filtered_tokens = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            filtered_tokens.append(token.lemma_)
            
        return " ".join(filtered_tokens)
    
    def load_model(self):

        """
        Function 1: Loads the trained model execution flow (Pipeline) from the file.
        """
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.label_encoder = data['encoder']
        else:
            print("No existing model found. Please run train_model() first.")
            self.model = None
            self.label_encoder = None

    # --- FUNCTION 2: TRAIN MODEL (With Sentence Transformer & Config) ---
    def train_model(self, taxonomy_file = 'taxonomy.json'):

        """
        Function 2: Handles category mapping using Sentence Transformer and trains the model.
        :param new_categories_config: List of strings (new categories) from the user's JSON config.
        """
        print("Starting training process...")
        
        # 1. Load Dataset
        df = pd.read_csv(self.data_path)

        if os.path.exists(self.extra_data_path):
            print(f"Loading extra data from {self.extra_data_path}...")
            df_extra = pd.read_csv(self.extra_data_path)
            df = pd.concat([df, df_extra], ignore_index=True)
            print(f"Total records after adding extra data: {len(df)}")
            
        if 'original_category' not in df.columns:
            print("Creating 'original_category' backup column...")
            df['original_category'] = df['category']
            # Save immediately so we don't lose this structure
            df.to_csv(self.data_path, index=False)
 
        if os.path.exists(self.feedback_path):
            print(f"Found feedback data in {self.feedback_path}. Merging...")
            df_fb = pd.read_csv(self.feedback_path)
            if not df_fb.empty:
                # specific columns to ensure match
                df_fb['original_category'] = df_fb['category']
                df_fb = df_fb[['transaction_description', 'currency', 'category','original_category']]
                df = pd.concat([df, df_fb], ignore_index=True)
                print(f"Training on {len(df)} records (Original + Feedback).")

        # 2. Sentence Transformer Mapping (If user provided new config)
        if taxonomy_file and os.path.exists(taxonomy_file):
            print(f"Loading User Taxonomy from {taxonomy_file}...")
            
            # 1. Load the JSON list
            with open(taxonomy_file, 'r') as f:
                new_categories_config = json.load(f)
            
            print(f"Target Categories: {new_categories_config}")
            
            print("Mapping old labels to new user configuration...")
            
            # Load Sentence Transformer (Deep Learning)
            mapper_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            
            old_categories = df["original_category"].unique()
            new_categories = new_categories_config

            # Encode both sets of labels
            old_embs = mapper_model.encode(old_categories, normalize_embeddings=True)
            new_embs = mapper_model.encode(new_categories, normalize_embeddings=True)

            # Calculate Cosine Similarity to find best matches
            def build_semantic_mapping(old_cats, new_cats, old_embs, new_embs, threshold=0.4):

                mapping = {}
                
                # cosine similarity = dot product because we normalized embeddings
                sim_matrix = np.matmul(old_embs, new_embs.T)  # shape: [len(old), len(new)]
                
                for i, old_cat in enumerate(old_cats):
                    sims = sim_matrix[i]                  # similarity to every new category
                    best_idx = np.argmax(sims)            # index of best new category
                    best_score = sims[best_idx]
                    best_new_cat = new_cats[best_idx]
                    
                    if best_score >= threshold:
                        mapping[old_cat] = {
                            "mapped_to": best_new_cat,
                            "score": float(best_score), 
                        }
                    else:
                        # low confidence: keep original or flag for manual review
                        mapping[old_cat] = {
                            "mapped_to": old_cat,
                            "score": float(best_score),
                            "low_confidence": True,
                        }
                
                return mapping

            mapping_dict = build_semantic_mapping(old_categories, new_categories, old_embs, new_embs, threshold=0.4)


            def map_category(cat, mapping):
                info = mapping.get(cat)
                if info is None:
                    return cat  # fallback
                return info["mapped_to"]

            # Apply mapping to dataset
            df["mapped_category"] = df["original_category"].apply(lambda c: map_category(c, mapping_dict))
            
            # Save the updated dataset (optional, to persist the mapping)
            df["category"] = df["mapped_category"]
            df = df.drop(columns = ['mapped_category'])
            df.to_csv(self.data_path, index=False)
            print(f"Category mapping complete: {mapping_dict}")
        
        else:
            print("No taxonomy file found. Using existing dataset categories.")

        # 3. Create Pipeline (Vectorizer + Classifier)
        # We embed the vectorizer in the model file so we don't need separate files
        preprocessor = ColumnTransformer(
            transformers=[
                ('tfidf', TfidfVectorizer(max_features=10000), 'transaction_description'),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['currency'])
            ],
            remainder='passthrough'
        )

        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MultinomialNB())
        ])

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['category'])   
        X = df.drop(columns=['category','original_category'])

        # 4. Train and Save
        clf.fit(X, y)
        joblib.dump({'model': clf, 'encoder': label_encoder}, self.model_path)
        self.model = clf
        self.label_encoder = label_encoder 
        print("Model trained and saved successfully.")
    
    
    def predict_transaction(self, description, currency="INR"):

        """
        Function 3: Predicts category and provides LIME explanation.
        """
        if not self.model or not self.label_encoder:
            return {"error": "Model or Encoder not loaded"}

        # --- 1. NORMALIZE INPUTS ---
        # Check if input is a list or a single string
        is_batch = isinstance(description, list)
        
        # Convert single string to list so we can process everything uniformly
        desc_list = description if is_batch else [description]
        
        clean_desc_list = [self._preprocess(text) for text in desc_list]

        # Handle Currency (Broadcast single currency if needed)
        if isinstance(currency, list):
            curr_list = currency
            if len(curr_list) != len(clean_desc_list):
                return {"error": "Description and Currency list lengths do not match"}
        else:
            curr_list = [currency] * len(clean_desc_list)

        # --- 2. FAST VECTORIZED PREDICTION ---
        # We predict everything in one shot using Matrix operations
        input_data = pd.DataFrame({
            'transaction_description': clean_desc_list,
            'currency': curr_list
        })

        # Get indices and probabilities for the whole batch
        pred_indices = self.model.predict(input_data)
        probs = self.model.predict_proba(input_data)
        
        # Decode all labels at once
        pred_labels = self.label_encoder.inverse_transform(pred_indices)

        # --- 3. GENERATE RESULTS & LIME EXPLANATIONS ---
        results = []
        
        # Initialize Explainer (Do this only once to save time)
        explainer = LimeTextExplainer(class_names=self.label_encoder.classes_)

        for i in range(len(clean_desc_list)):
            current_desc = clean_desc_list[i]
            current_curr = curr_list[i]
            current_prob = probs[i][pred_indices[i]]
            current_label = pred_labels[i]

            # --- LIME Logic (Per Transaction) ---
            # We define a wrapper specifically for THIS transaction's context
            def lime_predict_wrapper(text_list):
                # Helper to format data exactly how the model expects it
                df_lime = pd.DataFrame({
                    "transaction_description": text_list,
                    "currency": [current_curr] * len(text_list)
                })
                return self.model.predict_proba(df_lime)

            try:
                # Run LIME
                exp = explainer.explain_instance(
                    text_instance=current_desc, 
                    classifier_fn=lime_predict_wrapper, 
                    num_features=5,
                    top_labels=1
                )
                explanation_list = exp.as_list(label=exp.top_labels[0])
            except Exception:
                explanation_list = []

            # Append result
            results.append({
                "transaction": current_desc,
                "currency": current_curr,
                "predicted_category": current_label,
                "confidence": float(current_prob),
                "explanation": explanation_list
            })

        # --- 4. RETURN FORMAT ---
        # If user sent a list, return a list. If user sent a string, return a single dict.
        if is_batch:
            return results
        else:
            return results[0]
    # --- FUNCTION 3: PREDICT (Unified: Single or Batch) ---
    def predict_batch_transaction(self, description, currency="INR"):
        """
        Function 3: Predicts category and provides LIME explanation.
        Accepts either single strings OR lists.
        
        :param description: String "Uber" OR List ["Uber", "Netflix"]
        :param currency: String "USD" OR List ["USD", "USD"]
        :return: Dictionary (if single input) OR List of Dictionaries (if list input)
        """
        if not self.model or not self.label_encoder:
            return {"error": "Model or Encoder not loaded"}

        # --- 1. NORMALIZE INPUTS ---
        # Check if input is a list or a single string
        is_batch = isinstance(description, list)
        
        # Convert single string to list so we can process everything uniformly
        desc_list = description if is_batch else [description]
        
        clean_desc_list = [self._preprocess(text) for text in desc_list]

        # Handle Currency (Broadcast single currency if needed)
        if isinstance(currency, list):
            curr_list = currency
            if len(curr_list) != len(clean_desc_list):
                return {"error": "Description and Currency list lengths do not match"}
        else:
            curr_list = [currency] * len(clean_desc_list)

        # --- 2. FAST VECTORIZED PREDICTION ---
        # We predict everything in one shot using Matrix operations
        input_data = pd.DataFrame({
            'transaction_description': clean_desc_list,
            'currency': curr_list
        })

        # Get indices and probabilities for the whole batch
        pred_indices = self.model.predict(input_data)
        probs = self.model.predict_proba(input_data)
        
        # Decode all labels at once
        pred_labels = self.label_encoder.inverse_transform(pred_indices)

        # --- 3. GENERATE RESULTS & LIME EXPLANATIONS ---
        results = []
        
        
        # Append result
        for i in range(len(pred_indices)):
            results.append({
                "transaction": clean_desc_list[i],
                "currency": curr_list[i],
                "predicted_category": pred_labels[i],
                "confidence": float(probs[i][pred_indices[i]]),
            })
        # --- 4. RETURN FORMAT ---
    
        return results
    
    # --- FxUNCTION 4: ADD FEEDBACK (Human Loop & Retraining) ---
    def add_feedback(self, description, currency, correct_category):
        """
        Stores feedback. If threshold is met, saves to CSV and Re-trains.
        """
        print(f"Feedback received: {description} -> {correct_category}")
        
        new_row = pd.DataFrame([{
            'transaction_description': description,
            'currency': currency,
            'category': correct_category,
            'original_category': correct_category
        }])
    
        # 1. SAVE IMMEDIATELY (Safety First!)
        # We append to the file so we don't lose data on restart
        if not os.path.exists(self.feedback_path):
            new_row.to_csv(self.feedback_path, index=False)
        else:
            new_row.to_csv(self.feedback_path, mode='a', header=False, index=False)

        # 3. Check how many records are currently in the storage
        # We read the file to count rows. 
        try:
            current_feedback_df = pd.read_csv(self.feedback_path)
            current_count = len(current_feedback_df)
        except Exception:
            current_count = 0

        # 4. Check Threshold
        # If we have reached (or passed) the threshold, we retrain.
        if current_count > 0 and current_count % self.feedback_threshold == 0:
            print(f"Threshold ({self.feedback_threshold}) reached. Total feedback rows: {current_count}. Retraining...")
            
            # Trigger Retraining (This merges CSV + Feedback and rebuilds model)
            self.train_model()
            
            # OPTIONAL: Clear the feedback file after successful training?
            # If your train_model() merges and SAVES the combined data to transactions.csv, you should clear this.
            # If your train_model() only reads both files but doesn't save the merge, DO NOT CLEAR THIS.
            # Based on your current code, you DO NOT save the merge, so we keep the file.
            
            return {"status": "Model Retrained", "total_feedback": current_count}
        
        else:
            remaining = self.feedback_threshold - (current_count % self.feedback_threshold)
            print(f"Feedback saved. {remaining} more needed for retrain.")
            return {
                "status": "Feedback Buffered", 
                "remaining_until_retrain": remaining
            }