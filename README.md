Transaction Categorization SDK

A machine-learning-powered SDK for classifying financial transactions (e.g., "Starbucks" → "Food & Dining"). It features Semantic Taxonomy Mapping, Explainability (LIME), and a Self-Learning Feedback Loop.

🚀 Key Features

Batch & Single Prediction: Optimized functions for real-time UI display (single) or bulk statement processing (batch).
Dynamic Taxonomy: configurable via taxonomy.json. The model uses Sentence Transformers to map historical data to new user-defined categories without losing data integrity.
Explainability: Returns "Confidence Scores" and "Contributing Words" (via LIME) to explain why a transaction was categorized a certain way.
Human-in-the-Loop: An automated feedback system that collects user corrections and retrains the model periodically.

🛠️ Installation & Setup


1. Dependencies

Ensure you have Python 3.9+ installed. Install the required libraries:

Bash


pip install numpy pandas scikit-learn sentence-transformers lime spacy joblib



2. Download NLP Models

The SDK uses Spacy for text cleaning and Sentence Transformers for category mapping.

Bash


# Download Spacy English core
python -m spacy download en_core_web_sm


(Note: The SDK attempts to download this automatically on first run, but manual installation is recommended for production environments.)

3. Project Structure

Ensure your directory looks like this:

Plaintext


project_root/
│
├── main.py                 # The SDK Source Code
├── taxonomy.json           # User-defined categories
├── transactions.csv        # The training dataset
├── model.pkl               # (Generated after training)
└── feedback_storage.csv    # (Generated automatically)



⚙️ Configuration: Taxonomy

You can define the output categories in taxonomy.json. The model will semantically map existing data to these new labels during training.
Default taxonomy.json:

JSON


[
    "Utilities & Services",
    "Government & Legal",
    "Financial Services",
    "Income",
    "Charity & Donations",
    "Shopping & Retail",
    "Healthcare & Medical",
    "Entertainment & Recreation",
    "Transportation",
    "Food & Dining"
]



📖 API Usage


1. Initialization


Python


from main import TransactionModelSDK

# Initialize SDK (Loads model.pkl automatically)
# feedback_threshold=10 means the model retrains every 10 feedback items.
sdk = TransactionModelSDK(feedback_threshold=10)



2. Training the Model

Run this once to initialize the model or manually force a retrain.

Python


# Trains using transactions.csv and maps to categories in taxonomy.json
sdk.train_model(taxonomy_file='taxonomy.json')



3. Prediction (Single Transaction)

Use this for real-time UI requests. It includes LIME explanations (slower).

Python


txn_desc = "Uber trip to office"
currency = "USD"

result = sdk.predict_transaction(txn_desc, currency)

print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']}")
print(f"Reasoning: {result['explanation']}") 
# Output: [('Uber', 0.45), ('trip', 0.10)]



4. Prediction (Batch Processing)

Use this for processing bank statements or CSV uploads. It skips LIME for high performance.

Python


descriptions = ["Netflix Sub", "Shell Gas", "Salary"]
currencies = ["USD", "USD", "EUR"]

results = sdk.predict_batch_transaction(descriptions, currencies)

for res in results:
    print(f"{res['transaction']} -> {res['predicted_category']}")



🔄 The Feedback Loop (Self-Learning)

The SDK implements a Non-Destructive Feedback Loop. It does not modify the original transactions.csv immediately but uses a separate storage file.

Workflow:

User Corrects a Transaction: The frontend sends the correct category.
SDK Buffers Data: add_feedback() saves it to feedback_storage.csv.
Threshold Check:
If feedback_count % threshold == 0 (e.g., 10th, 20th, 30th record), the SDK triggers train_model().
The model retrains on transactions.csv + feedback_storage.csv.

Implementation:


Python


# User corrects "Starbucks" from "Shopping" to "Food & Dining"
response = sdk.add_feedback(
    description="Starbucks coffee", 
    currency="USD", 
    correct_category="Food & Dining"
)

print(response)
# Output: {'status': 'Feedback Buffered', 'remaining_until_retrain': 9}
# OR
# Output: {'status': 'Model Retrained', 'total_feedback': 10}



🏗️ Architecture Notes for Developers


Data Integrity ("Source of Truth")

The SDK creates a backup column original_category in your dataset during the first run.
Why? If a user changes the taxonomy to something broad (e.g., "Expenses"), we lose detail.
Benefit: If the user later changes the taxonomy back to detailed categories, the SDK maps from the original_category backup, preserving historical accuracy.

Performance Tips

Batch vs Single: Always use predict_batch_transaction for lists. It uses vectorized matrix operations and is ~100x faster than looping predict_transaction.
Model Storage: The model.pkl file includes both the classifier and the label encoder. Do not delete it unless you want to force a full retrain from scratch.

❓ Troubleshooting

Error: Model or Encoder not loaded
Ensure model.pkl exists. If not, run sdk.train_model() first.
Error: Description and Currency list lengths do not match
When using batch prediction, ensure the list of currencies matches the list of descriptions, or pass a single currency string to broadcast it.
New Category Error
If add_feedback introduces a category NOT in taxonomy.json, the model will learn it, but it is recommended to update taxonomy.json to match the new reality.
