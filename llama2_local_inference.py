# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================

import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import warnings

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings('ignore')

print("="*70)
print("Llama-2-7B Local Inference Script")
print("="*70)


# ============================================================================
# STEP 2: Device Setup - Automatically Detect GPU or CPU
# ============================================================================

def setup_device():
    """
    Automatically detect the best available device for inference.
    
    Priority:
    1. CUDA (NVIDIA GPU) - Fastest
    2. MPS (Apple Silicon M1/M2/M3) - Fast on Mac
    3. CPU - Slowest but works everywhere
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✅ GPU Detected: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        print(f"   Using CUDA for inference")
        
        # Check if GPU memory is sufficient
        if gpu_memory < 8:
            print("\n⚠️  WARNING: GPU has less than 8GB memory.")
            print("   Model will use 4-bit quantization to fit.")
        
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print(f"\n✅ Apple Silicon Detected (M1/M2/M3)")
        print(f"   Using MPS (Metal Performance Shaders) for inference")
        
    else:
        device = "cpu"
        print(f"\n⚠️  No GPU detected - Using CPU")
        print("   WARNING: Inference will be slow on CPU (30+ seconds per response)")
        print("   Consider using a GPU for better performance")
    
    return device


# ============================================================================
# STEP 3: Model and Tokenizer Loading
# ============================================================================

def load_model_and_tokenizer(model_name="meta-llama/Llama-2-7b", device="cuda", use_4bit=True):
    """
    Load the Llama-2-7b model and tokenizer from Hugging Face.
    
    This function handles:
    - Automatic quantization (4-bit or 8-bit) to reduce memory usage
    - Token authentication for gated models
    - Error handling for common issues
    
    Args:
        model_name (str): HuggingFace model identifier
        device (str): Device to load model on ('cuda', 'mps', 'cpu')
        use_4bit (bool): Use 4-bit quantization to reduce memory (CUDA only)
    
    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    print(f"\n{'='*70}")
    print("Loading Model and Tokenizer")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"4-bit quantization: {use_4bit and device == 'cuda'}")
    
    # Check for HuggingFace token (required for Llama-2)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        print("\n⚠️  WARNING: HF_TOKEN not found in environment variables.")
        print("   Llama-2 is a gated model that requires authentication.")
        print("\n   To fix this:")
        print("   1. Get a token from https://huggingface.co/settings/tokens")
        print("   2. Accept Llama-2 license at https://huggingface.co/meta-llama/Llama-2-7b")
        print("   3. Set token: export HF_TOKEN='your_token_here'")
        print("\n   Attempting to load without token (may fail)...")
    
    try:
        # ====================================================================
        # Load Tokenizer
        # ====================================================================
        print("\n📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True
        )
        
        # Set padding token (required for batch processing)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print("✅ Tokenizer loaded successfully")
        
        # ====================================================================
        # Load Model with Quantization (if GPU available)
        # ====================================================================
        print("\n📥 Loading model (this may take 2-5 minutes on first run)...")
        
        # Configure quantization for memory efficiency
        if use_4bit and device == "cuda":
            # 4-bit quantization: Reduces memory from ~28GB to ~7GB
            print("   Using 4-bit quantization (NF4) to reduce memory usage")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # Nested quantization
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",  # Automatically distribute across GPUs
                token=hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        
        elif device == "cuda":
            # 8-bit quantization (fallback if 4-bit not requested)
            print("   Using 8-bit quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True
            )
        
        else:
            # CPU or MPS - no quantization
            print("   Loading in full precision (no quantization)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        
        print("✅ Model loaded successfully")
        print(f"   Model size: ~{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
        
        return model, tokenizer
    
    except torch.cuda.OutOfMemoryError:
        print("\n❌ ERROR: CUDA Out of Memory!")
        print("   Your GPU doesn't have enough VRAM for this model.")
        print("\n   Solutions:")
        print("   1. Close other GPU applications")
        print("   2. Use 4-bit quantization (already enabled)")
        print("   3. Use a smaller model (e.g., Llama-2-7b-chat)")
        print("   4. Use CPU (will be slower)")
        return None, None
    
    except Exception as e:
        print(f"\n❌ ERROR loading model: {str(e)}")
        print("\n   Common issues:")
        print("   - Missing HF_TOKEN for gated models")
        print("   - Insufficient disk space (~15GB needed)")
        print("   - Network connection issues")
        return None, None


# ============================================================================
# STEP 4: Text Generation Function
# ============================================================================

def generate_response(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    device="cuda"
):
    """
    Generate a response from the model given a prompt.
    
    Args:
        model: The loaded Llama model
        tokenizer: The loaded tokenizer
        prompt (str): Input text prompt
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Controls randomness (0.0-1.0, higher=more random)
        top_p (float): Nucleus sampling threshold
        device (str): Device the model is on
    
    Returns:
        str: Generated text response
    """
    print(f"\n{'='*70}")
    print("Generating Response")
    print(f"{'='*70}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"{'='*70}\n")
    
    try:
        # Tokenize the input prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the same device as model
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        print("🤖 Generating...")
        
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,  # Enable sampling for diverse outputs
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1  # Reduce repetition
            )
        
        # Decode the generated tokens to text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the newly generated part (remove the prompt)
        response = generated_text[len(prompt):].strip()
        
        return response
    
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        return None


# ============================================================================
# STEP 5: Main Function - Putting It All Together
# ============================================================================

def main():
    """
    Main function that orchestrates the entire inference pipeline.
    """
    
    # Step 1: Setup device
    device = setup_device()
    
    # Step 2: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name="meta-llama/Llama-2-7b",
        device=device,
        use_4bit=True  # Enable 4-bit quantization for lower memory usage
    )
    
    # Check if model loaded successfully
    if model is None or tokenizer is None:
        print("\n❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("✅ Model Ready for Inference!")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # Example 1: Medical/Biological Question
    # ========================================================================
    
    prompt1 = "Explain what gene expression is in simple terms."
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Gene Expression Explanation")
    print("="*70)
    
    response1 = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt1,
        max_new_tokens=200,
        temperature=0.7,
        device=device
    )
    
    if response1:
        print("\n📤 Response:")
        print("-" * 70)
        print(response1)
        print("-" * 70)
    
    # ========================================================================
    # Example 2: Medical Question
    # ========================================================================
    
    prompt2 = "What are the main symptoms of Type 2 Diabetes?"
    
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Diabetes Symptoms")
    print("="*70)
    
    response2 = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt2,
        max_new_tokens=150,
        temperature=0.6,
        device=device
    )
    
    if response2:
        print("\n📤 Response:")
        print("-" * 70)
        print(response2)
        print("-" * 70)
    
    # ========================================================================
    # Interactive Mode (Optional - Uncomment to enable)
    # ========================================================================
    
    print("\n\n" + "="*70)
    print("Want to try your own prompts? (y/n): ", end="")
    
    try:
        user_choice = input().strip().lower()
        
        if user_choice == 'y':
            print("\n" + "="*70)
            print("Interactive Mode - Type 'quit' to exit")
            print("="*70)
            
            while True:
                print("\n💬 Your prompt: ", end="")
                user_prompt = input().strip()
                
                if user_prompt.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_prompt:
                    print("⚠️  Empty prompt. Please enter a question.")
                    continue
                
                user_response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=user_prompt,
                    max_new_tokens=200,
                    temperature=0.7,
                    device=device
                )
                
                if user_response:
                    print("\n🤖 Response:")
                    print("-" * 70)
                    print(user_response)
                    print("-" * 70)
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting.")
    
    print("\n" + "="*70)
    print("Script completed successfully!")
    print("="*70)


# ============================================================================
# STEP 6: Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Entry point of the script.
    
    To run this script:
    1. Install dependencies: pip install -r requirements.txt
    2. Set HF token: export HF_TOKEN='your_token_here'
    3. Accept Llama-2 license: https://huggingface.co/meta-llama/Llama-2-7b
    4. Run: python llama2_local_inference.py
    """
    main()

