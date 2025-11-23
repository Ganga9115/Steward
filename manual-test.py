from main import TransactionModelSDK
import os

def run_manual_test():
 
    print("--- 1. INITIALIZING SDK ---")
    # We set a low threshold (2) so we can trigger retraining quickly
    sdk = TransactionModelSDK(feedback_threshold=2)

    # sdk.train_model(new_categories_config=['Subscriptions & Entertainment', 'Utilities'])

    if not os.path.exists("model.pkl"):
        sdk.train_model()

    print("\n--- 3. PREDICTING ---")
    # Test a prediction
    txn_desc = "Bitcoin purchase Binance"
    txn_curr='USD'
    result = sdk.predict_transaction(description=txn_desc, currency=txn_curr)
    # print(f"Input: '{new_transaction}'")
    print("\n" + "="*30)
    print(f" Predicted Category : {result['predicted_category']}")
    print(f" Probability        : {result['confidence']:.4f}")
    print("="*30)
    print(" Top contributing words:")
    for word, score in result['explanation']:
        # This prints: "Court                 +0.2033"
        print(f"{word:20s}  {score:+.4f}")
    print("="*30 + "\n")


    print("\n--- 3. GIVING FEEDBACK ---")
    # Simulating user saying: "No, that is Investment"
    # Note: "Investment" might be a NEW category not in the original dataset!
    
    # Feedback 1
    sdk.add_feedback(txn_desc, txn_curr, "Investment") 
    
    # Feedback 2 (Triggers Retrain because threshold is 2)
    sdk.add_feedback("Coinbase deposit", "USD", "Investment")

    print("\n--- 4. PREDICTING (After Feedback) ---")
    # Now the model should know "Investment" exists
    res_new = sdk.predict_transaction(txn_desc, txn_curr)
    print("\n" + "="*30)
    print(f" Predicted Category : {res_new['predicted_category']}")
    print(f" Probability        : {res_new['confidence']:.4f}")
    print("="*30)
    print(" Top contributing words:")
    for word, score in res_new['explanation']:
        # This prints: "Court                 +0.2033"
        print(f"{word:20s}  {score:+.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    run_manual_test()