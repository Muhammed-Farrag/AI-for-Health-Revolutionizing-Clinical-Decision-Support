# Stitch just created this file to test the setup of the llama inference script.
import os
from llama_inference_script import LlamaInferenceProcessor


# ============================================================================
# CONFIGURATION - Update these when you have your dataset
# ============================================================================
DATASET_PATH = "your_dataset.csv"  # TODO: Replace with your actual dataset path
PROMPT_COLUMN = "question"         # TODO: Replace with your column name containing prompts
OUTPUT_PATH = "results_output.csv" # Where to save results
SYSTEM_PROMPT = "You are a helpful medical AI assistant for clinical decision support."


def test_connection():
    """Test 1: Verify API connection and basic setup."""
    print("\n" + "="*60)
    print("TEST 1: Connection & Setup")
    print("="*60 + "\n")
    
    # Check if HF_TOKEN is set
    if not os.environ.get("HF_TOKEN"):
        print("❌ ERROR: HF_TOKEN environment variable not set!")
        print("\nPlease set your Hugging Face token:")
        print("   export HF_TOKEN='your_token_here'")
        return False
    
    print("✓ HF_TOKEN environment variable is set")
    
    try:
        # Initialize processor
        processor = LlamaInferenceProcessor()
        print("✓ Successfully initialized LlamaInferenceProcessor")
        
        # Test a simple query to verify connection
        print("\n📤 Testing API connection with simple query...")
        response = processor.query_single(
            user_prompt="Respond with 'Connection successful' if you receive this.",
            temperature=0.3,
            max_tokens=20
        )
        
        print(f"📥 Response: {response}")
        print("\n✅ Connection test PASSED!\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Connection test FAILED: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify your HF_TOKEN is correct")
        print("2. Check your internet connection")
        print("3. Ensure you have access to the LLaMA model on Hugging Face")
        return False


def test_medical_query():
    """Test 2: Test with a sample medical query to verify model responses."""
    print("\n" + "="*60)
    print("TEST 2: Medical Query Test")
    print("="*60 + "\n")
    
    try:
        processor = LlamaInferenceProcessor()
        
        # Test with a medical question
        print("📤 Asking medical question...")
        response = processor.query_single(
            user_prompt="What are the main symptoms of Type 2 Diabetes?",
            system_prompt=SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=200
        )
        
        print(f"\n📥 Response:\n{response}\n")
        print("✅ Medical query test PASSED!\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Medical query test FAILED: {str(e)}\n")
        return False


def test_dataset_processing():
    """Test 3: Process your custom dataset (when available)."""
    print("\n" + "="*60)
    print("TEST 3: Dataset Processing")
    print("="*60 + "\n")
    
    # Check if dataset file exists
    if not os.path.exists(DATASET_PATH):
        print(f"⚠️  Dataset not found: {DATASET_PATH}")
        print("\nThis test will run when you:")
        print("1. Place your dataset file in the project directory")
        print("2. Update DATASET_PATH at the top of this file")
        print("3. Update PROMPT_COLUMN to match your dataset's column name")
        print("\nSkipping dataset test for now...\n")
        return None
    
    try:
        processor = LlamaInferenceProcessor()
        
        print(f"📂 Processing dataset: {DATASET_PATH}")
        print(f"📋 Using column: {PROMPT_COLUMN}")
        print(f"💾 Output will be saved to: {OUTPUT_PATH}\n")
        
        # Process the dataset
        results_df = processor.process_dataset(
            dataset_path=DATASET_PATH,
            prompt_column=PROMPT_COLUMN,
            output_path=OUTPUT_PATH,
            system_prompt=SYSTEM_PROMPT,
            batch_size=5  # Show progress every 5 rows
        )
        
        print(f"\n✅ Dataset processing PASSED!")
        print(f"📊 Processed {len(results_df)} rows")
        print(f"💾 Results saved to: {OUTPUT_PATH}\n")
        
        # Show first few results
        print("Preview of results:")
        print(results_df.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset processing FAILED: {str(e)}\n")
        return False


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*70)
    print(" "*15 + "🚀 LLaMA MODEL TEST SUITE 🚀")
    print("="*70)
    
    # Track results
    results = {}
    
    # Test 1: Connection
    results['connection'] = test_connection()
    
    if not results['connection']:
        print("\n⚠️  Stopping tests - fix connection issues first.\n")
        return
    
    # Test 2: Medical Query
    results['medical_query'] = test_medical_query()
    
    # Test 3: Dataset Processing (optional - runs when dataset is available)
    results['dataset'] = test_dataset_processing()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Connection Test:       {'✅ PASSED' if results['connection'] else '❌ FAILED'}")
    print(f"Medical Query Test:    {'✅ PASSED' if results['medical_query'] else '❌ FAILED'}")
    
    if results['dataset'] is None:
        print(f"Dataset Processing:    ⏭️  SKIPPED (no dataset provided yet)")
    else:
        print(f"Dataset Processing:    {'✅ PASSED' if results['dataset'] else '❌ FAILED'}")
    
    print("="*70 + "\n")
    
    # Final message
    if results['connection'] and results['medical_query']:
        if results['dataset'] is None:
            print("🎉 Setup is complete and working!")
            print("\n📝 NEXT STEPS FOR YOUR FRIEND:")
            print("1. Place the dataset file in this directory")
            print("2. Open this file (test_llama.py)")
            print("3. Update these variables at the top:")
            print(f"   - DATASET_PATH = '{DATASET_PATH}'  # Change to actual filename")
            print(f"   - PROMPT_COLUMN = '{PROMPT_COLUMN}'  # Change to actual column name")
            print("4. Run this test again: python test_llama.py")
            print("\n✨ The script will then process the full dataset!\n")
        elif results['dataset']:
            print("🎉 All tests passed! Your dataset has been processed successfully!\n")
    else:
        print("⚠️  Some tests failed. Please fix the issues above before proceeding.\n")


if __name__ == "__main__":
    run_all_tests()

