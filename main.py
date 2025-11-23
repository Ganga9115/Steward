import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer


class TransactionModelSDK:
    def __init__(self, model_path='model.pkl', data_path='transactions.csv', feedback_path = 'feedback_storage.csv', feedback_threshold=10):

        """
        Initialize the SDK.
        :param model_path: Where the trained model is saved.
        :param data_path: Where the dataset is stored.
        :param feedback_threshold: How many feedback items to wait for before retraining.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.feedback_path = feedback_path
        self.feedback_threshold = feedback_threshold
        self.feedback_buffer = []  # Temporary storage for feedback
        self.model = None
        self.label_encoder = None

        # Load the model immediately if it exists
        self.load_model()

    # --- FUNCTION 1: LOAD MODEL ---
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
    def train_model(self, new_categories_config=None):

        """
        Function 2: Handles category mapping using Sentence Transformer and trains the model.
        :param new_categories_config: List of strings (new categories) from the user's JSON config.
        """
        print("Starting training process...")
        
        # 1. Load Dataset
        df = pd.read_csv(self.data_path)

        if os.path.exists(self.feedback_path):
            print(f"Found feedback data in {self.feedback_path}. Merging...")
            df_fb = pd.read_csv(self.feedback_path)
            if not df_fb.empty:
                # specific columns to ensure match
                df_fb = df_fb[['transaction_description', 'currency', 'category']]
                df = pd.concat([df, df_fb], ignore_index=True)
                print(f"Training on {len(df)} records (Original + Feedback).")

        y = df['category']

        # 2. Sentence Transformer Mapping (If user provided new config)
        if new_categories_config:
            
            print("Mapping old labels to new user configuration...")
            
            # Load Sentence Transformer (Deep Learning)
            mapper_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            
            old_categories = y.unique()
            new_categories = new_categories_config

            # Encode both sets of labels
            old_embs = mapper_model.encode(old_categories, normalize_embeddings=True)
            new_embs = mapper_model.encode(new_categories, normalize_embeddings=True)

            # Calculate Cosine Similarity to find best matches
            def build_semantic_mapping(old_cats, new_cats, old_embs, new_embs, threshold=0.5):

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
            df["mapped_category"] = df["category"].apply(lambda c: map_category(c, mapping_dict))
            
            # Save the updated dataset (optional, to persist the mapping)
            df["category"] = df["mapped_category"]
            df = df.drop(columns = ['mapped_category'])
            df.to_csv(self.data_path, index=False)
            print(f"Category mapping complete: {mapping_dict}")

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
        X = df.drop(columns=['category'])

        # 4. Train and Save
        clf.fit(X, y)
        joblib.dump({'model': clf, 'encoder': label_encoder}, self.model_path)
        self.model = clf
        self.label_encoder = label_encoder 
        print("Model trained and saved successfully.")

    # --- FUNCTION 3: PREDICT (With LIME Explainability) ---
    def predict_transaction(self, description, currency):
        """
        Function 3: Predicts category and provides LIME explanation.
        :return: Dictionary containing prediction and explanation list.
        """
        if not self.model or not self.label_encoder:
            return {"error": "Model or Encoder not loaded"}

        input_data = pd.DataFrame(
            [[description, currency]], 
            columns=['transaction_description', 'currency']
        )
        # 1. Make Prediction
        pred_idx = self.model.predict(input_data)[0]
        probs = self.model.predict_proba(input_data)[0]
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx]
        # probabilities = self.model.predict_proba(transaction_text)

        # 2. Generate LIME Explanation
        def lime_predict_wrapper(text_list):
            # Create a DataFrame where 'description' varies, but 'currency' stays constant
            df_lime = pd.DataFrame({
                "transaction_description": text_list,
                "currency": [currency] * len(text_list) # Repeat currency for every text variation
            })
            return self.model.predict_proba(df_lime)

        # Initialize LIME Explainer with your specific class names
        explainer = LimeTextExplainer(class_names=self.label_encoder.classes_)

        # Generate Explanation
        # num_features=5 means "Show me the top 5 words that caused this decision"
        exp = explainer.explain_instance(
            text_instance=description, 
            classifier_fn=lime_predict_wrapper, 
            num_features=5,
            top_labels=1
        )
        
        # Extract the explanation list: [('Court', 0.20), ('Fee', 0.05)]
        # exp.top_labels[0] ensures we get reasons for the PREDICTED category
        explanation_list = exp.as_list(label=exp.top_labels[0])


        return {
            "transaction": description,
            "currency": currency,
            "predicted_category": pred_label,
            "confidence": float(confidence),
            "explanation": explanation_list
        }

    # --- FxUNCTION 4: ADD FEEDBACK (Human Loop & Retraining) ---
    def add_feedback(self, description, currency, correct_category, weight = 50):
        """
        Stores feedback. If threshold is met, saves to CSV and Re-trains.
        """
        print(f"Feedback received: {description} -> {correct_category}")
        
        # Add to temporary buffer
        for _ in range(weight):
            self.feedback_buffer.append({
                'transaction_description': description,
                'currency': currency,
                'category': correct_category
            })

        # Check Threshold
        if len(self.feedback_buffer) >= self.feedback_threshold:
            print(f"Threshold ({self.feedback_threshold}) reached. Saving to {self.feedback_path} and retraining...")
            
            # 1. Convert buffer to DataFrame
            new_feedback_df = pd.DataFrame(self.feedback_buffer)
            
            # 2. Append to Feedback CSV (Create if not exists)
            if not os.path.exists(self.feedback_path):
                new_feedback_df.to_csv(self.feedback_path, index=False)
            else:
                new_feedback_df.to_csv(self.feedback_path, mode='a', header=False, index=False)
            
            # 3. Clear Buffer
            self.feedback_buffer = []
            
            # 4. Trigger Retraining 
            # (This will now load the updated feedback CSV automatically)
            self.train_model()
            
            return {"status": "Model Retrained", "buffer_size": 0}
        
        else:
            remaining = self.feedback_threshold - len(self.feedback_buffer)
            print(f"Feedback buffered. {remaining} more needed for retrain.")
            return {
                "status": "Feedback Buffered", 
                "remaining_until_retrain": remaining
            }

# ==========================================
# EXAMPLE USAGE (For the Developer)
# ==========================================
# if __name__ == "__main__":
#     # 1. Initialize
#     sdk = TransactionModelSDK(feedback_threshold=2) # Low threshold for testing

#     # 2. Developer passes a Config File (JSON list) to map categories
#     user_categories = ["Travel & Transport", "Food & Dining", "Utilities", "Income"]
#     # This triggers the Sentence Transformer mapping and initial training
#     sdk.train_model(new_categories_config=user_categories)

#     # 3. Make a Prediction (with LIME)
#     result = sdk.predict_transaction("Uber trip to office")
#     print("\nPrediction Result:")
#     print(result) 
#     # Output will show "Travel & Transport" and explain that "Uber" was the key reason.

#     # 4. Human Feedback Loop
#     # Let's say the user corrects a prediction
#     print("\n--- User gives feedback ---")
#     sdk.add_feedback("Starbucks coffee", "Food & Dining") # Buffer = 1
#     sdk.add_feedback("Monthly Rent payment", "Utilities")   # Buffer = 2 (Threshold hit!)
#     # The model automatically retrains here.