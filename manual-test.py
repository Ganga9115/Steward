from main import TransactionModelSDK
import os

def run_manual_test():
 
    print("--- 1. INITIALIZING SDK ---")
    # We set a low threshold (2) so we can trigger retraining quickly
    sdk = TransactionModelSDK(feedback_threshold=2)

    # sdk.train_model(new_categories_config=['Subscriptions & Entertainment', 'Utilities'])

    if not os.path.exists("model.pkl"):
        sdk.train_model(taxonomy_file='taxonomy.json')
        

    print("\n--- 3. PREDICTING ---")
    # Test a prediction
    txn_desc = "TicketNew Confirmation no is 241183/190929643 secretcode PCTIYDXY Vikram (U/A) - Tamil on Sun 05/06 08:10 AM FIRST CLASS J6 - 7 Tkts-2 Casino Theatre at Casino Cinemas RGB Laser 4K 3D A/C Dolby 7.1.Thank you"
    txn_curr='INR'
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




    # print("\n--- 3. GIVING FEEDBACK ---")
    # # Simulating user saying: "No, that is Investment"
    # # Note: "Investment" might be a NEW category not in the original dataset!
    
    # # Feedback 1
    # sdk.add_feedback(txn_desc, txn_curr, "Investment") 



    # # ==========================================
    # # TEST CASE 1: Single Transaction (String Input)
    # # ==========================================
    # print("\n" + "="*40)
    # print(" TEST 1: Single Input (String)")
    # print("="*40)
    
    # desc = "Uber Trip to Office"
    # curr = "USD"
    
    # # Call the function
    # result = sdk.predict_batch_transaction(desc, curr)
    
    # # Check type (Should be Dict)
    # print(f"Input:  {desc}")
    # print(f"Type:   {type(result)}") # Should say <class 'dict'>
    # print(f"Result: {result}")

    # # ==========================================
    # # TEST CASE 2: Batch Transactions (List Input)
    # # ==========================================
    # print("\n" + "="*40)
    # print(" TEST 2: Batch Input (List)")
    # print("="*40)

    # # Mixed list of inputs
    # batch_desc = [
    #     "Netflix Subscription", 
    #     "Shell Gas Station", 
    #     "Salary Credit", 
    #     "Murugan Idly Kadai"
    # ]
    # batch_curr = ["USD", "USD", "USD", "INR"]
    # # Call the function
    # batch_results = sdk.predict_batch_transaction(batch_desc, batch_curr)


    # # Check type (Should be List)
    # print(f"Input Size: {len(batch_desc)} items")
    # print(f"Output Type: {type(batch_results)}") # Should say <class 'list'>

    # # Print results nicely
    # print(f"{'TRANSACTION':<25} | {'PREDICTED CATEGORY':<20} | {'CONFIDENCE'}")
    # print("-" * 65)
    
    # for res in batch_results:
    #     print(f"{res['transaction']:<25} | {res['predicted_category']:<20} | {res['confidence']:.4f}")

if __name__ == "__main__":
    run_manual_test()