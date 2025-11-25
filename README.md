# ****Steward The Transaction Categorization SDK****

A machine-learning-powered SDK for classifying financial transactions (e.g., "Starbucks" → "Food & Dining"). It features Semantic Taxonomy Mapping, Explainability (LIME), and a Self-Learning Feedback Loop.

# 🚀 Key Features

Batch & Single Prediction: Optimized functions for real-time UI display (single) or bulk statement processing (batch).
Dynamic Taxonomy: configurable via taxonomy.json. The model uses Sentence Transformers to map historical data to new user-defined categories without losing data integrity.
Explainability: Returns "Confidence Scores" and "Contributing Words" (via LIME) to explain why a transaction was categorized a certain way.
Human-in-the-Loop: An automated feedback system that collects user corrections and retrains the model periodically.

# 🛠️ Installation & Setup


1. Dependencies

Ensure you have Python 3.9+ installed. Install the required libraries using requirements.txt file:

    Bash 
    pip install -r requirements.txt

2. Download NLP Models

The SDK uses Spacy for text cleaning and Sentence Transformers for category mapping.

    Bash 
    python -m spacy download en_core_web_sm

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
    └── extra_transactions.csv    # (If you want to add more data for training)



⚙️ Configuration: Taxonomy

You can define the output categories in taxonomy.json. The model will semantically map existing data to these new labels during training.
Default taxonomy.json:

    JSON [
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


1. Initialization


Python 

    from main import TransactionModelSDK
    
    # Initialize SDK (Loads model.pkl automatically)
    # feedback_threshold=10 means the model retrains every 10 feedback items.
    sdk = TransactionModelSDK(feedback_threshold=10)

# load_model
        1. **Purpose:** Loads the trained Machine Learning pipeline and Label Encoder from the `.pkl` file on disk. If the file is missing, it prints a warning.
        2. **Parameters:** None.
        3. **Output:** None (Updates `self.model` internally).


2. Training the Model

Run this once to initialize the model or manually force a retrain.

Python


    # Trains using transactions.csv and maps to categories in taxonomy.json
    sdk.train_model(taxonomy_file='taxonomy.json')


# train_model
        1. **Purpose:** The heavy lifter. It loads data (Main + Extra + Feedback), creates a backup "Source of Truth" column, maps categories using Sentence Transformers (if a taxonomy file is provided), builds             the TF-IDF pipeline, and trains the Naive Bayes classifier.
        2. Parameters:
            1. `taxonomy_file` (str): Path to the JSON file defining the allowed categories (Default: `'taxonomy.json'`).
        3. **Output:** None (Prints progress logs to console).


3. Prediction 

Use this for real-time UI requests. It includes LIME explanations (slower).

Python


    txn_desc = "Uber trip to office"
    currency = "USD"
    
    result = sdk.predict_transaction(txn_desc, currency)
    
    print(f"Category: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reasoning: {result['explanation']}") 
    # Output: [('Uber', 0.45), ('trip', 0.10)]
    
# `predict_transaction` (With Explainability)
        1. **Purpose:** Designed for **Single Real-Time Predictions** (e.g., User UI). It predicts the category and runs LIME to generate an explanation of *why* that category was chosen.
        2. Parameters:
            1. `description` (str OR list): The transaction text (e.g., `"Starbucks coffee"`).
            2. `currency` (str OR list): The currency code (e.g., `"USD"`).
        3. Output Format (JSON Dictionary):
            1. {
            "transaction": "Starbucks coffee",
            "currency": "USD",
            "predicted_category": "Food & Dining",
            "confidence": 0.854,
            "explanation": [
            ("Starbucks", 0.45),
            ("coffee", 0.20)
            ]
            }

4. Prediction (Batch Processing)

Use this for processing bank statements or CSV uploads. It skips LIME for high performance.

Python


    descriptions = ["Netflix Sub", "Shell Gas", "Salary"]
    currencies = ["USD", "USD", "EUR"]
    
    results = sdk.predict_batch_transaction(descriptions, currencies)
    
    for res in results:
        print(f"{res['transaction']} -> {res['predicted_category']}")

# `predict_batch_transaction` (High Performance)
        1. **Purpose:** Designed for **Bulk Processing** (e.g., CSV Uploads). It skips the slow LIME explanation step and uses vectorized operations to predict thousands of transactions in milliseconds.
        2. Parameters:
            1. `description` (list of strings): `["Netflix", "Uber", "Salary"]`
            2. `currency` (list of strings): `["USD", "USD", "EUR"]`
        3. Output Format (List of JSON Dictionaries):
            1. [
            {
            "transaction": "Netflix",
            "currency": "USD",
            "predicted_category": "Entertainment",
            "confidence": 0.99
            },
            {
            "transaction": "Uber",
            "currency": "USD",
            "predicted_category": "Travel",
            "confidence": 0.88
            }
            ]

🔄 The Feedback Loop (Self-Learning)

The SDK implements a Non-Destructive Feedback Loop. It does not modify the original transactions.csv immediately but uses a separate storage file called feedback storage.

# add_feedback
        1. **Purpose:** The "Human-in-the-Loop" engine. It saves the user's correction to `feedback_storage.csv` immediately. It checks if the total feedback count is a multiple of the `threshold` (e.g., 100,                 200, 300). If yes, it triggers `train_model()` automatically.
        2. Parameters:
            1. `description` (str): The transaction text.
            2. `currency` (str): The currency.
            3. `correct_category` (str): The actual category provided by the user.
        3. Output Format (Scenario A: Buffered):
            1. {
            "status": "Feedback Buffered",
            "remaining_until_retrain": 45
            }
        4. **Output Format (Scenario B: Retrained):**
            1. {
            "status": "Model Retrained",
            "total_feedback": 100
            }
            
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



# 🏗️ Architecture Notes for Developers


    # Data Integrity ("Source of Truth")
    
        The SDK creates a backup column original_category in your dataset during the first run.
        Why? If a user changes the taxonomy to something broad (e.g., "Expenses"), we lose detail.
        Benefit: If the user later changes the taxonomy back to detailed categories, the SDK maps from the original_category backup, preserving historical accuracy.
    
    # Performance Tips
    
        Batch vs Normal: Always use predict_batch_transaction for large lists when you don't use explainability.
        Model Storage: The model.pkl file includes both the classifier and the label encoder. Do not delete it unless you want to force a full retrain from scratch.
    
    # ❓ Troubleshooting
    
        Error: Model or Encoder not loaded
        Ensure model.pkl exists. If not, run sdk.train_model() first.
        Error: Description and Currency list lengths do not match
        When using batch prediction, ensure the list of currencies matches the list of descriptions, or pass a single currency string to broadcast it.
    New Category Error
    If add_feedback introduces a category NOT in taxonomy.json, the model will learn it, but it is recommended to update taxonomy.json to match the new reality.
