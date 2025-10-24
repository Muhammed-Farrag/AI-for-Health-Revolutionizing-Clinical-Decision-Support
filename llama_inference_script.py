import os
import json
from typing import List, Dict, Optional
from huggingface_hub import InferenceClient
import pandas as pd


class LlamaInferenceProcessor:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
       
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        
        if not self.api_key:
            raise ValueError(
                "No API key provided. Either pass api_key parameter or set HF_TOKEN environment variable."
            )
        
        # Store model name for reference
        self.model_name = model_name
        
        # Initialize the Hugging Face Inference Client
        # Provider is set to "novita" as per the provided example
        self.client = InferenceClient(
            provider="novita",
            api_key=self.api_key,
        )
        
        print(f"✓ Initialized LLaMA Inference Client with model: {self.model_name}")
    
    
    def query_single(
        self, 
        user_prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Send a single query to the LLaMA model and get the response.
        
        Args:
            user_prompt (str): The user's question or input text.
            system_prompt (str, optional): System message to set model behavior/context.
            temperature (float): Controls randomness (0.0-1.0). Lower = more focused.
            max_tokens (int): Maximum number of tokens in the response.
        
        Returns:
            str: The model's response text.
        
        Example:
            >>> processor = LlamaInferenceProcessor()
            >>> response = processor.query_single("What is diabetes?")
            >>> print(response)
        """
        # Build the messages list
        messages = []
        
        # Add system prompt if provided (useful for setting context or role)
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add the user's question/prompt
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        try:
            # Make the API call to LLaMA model
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract and return the response text
            response_text = completion.choices[0].message.content
            return response_text
            
        except Exception as e:
            print(f"❌ Error during inference: {str(e)}")
            return f"ERROR: {str(e)}"
    
    
    def process_dataset(
        self,
        dataset_path: str,
        prompt_column: str,
        output_path: str,
        system_prompt: Optional[str] = None,
        batch_size: int = 10
    ) -> pd.DataFrame:
        """
        Process an entire dataset through the LLaMA model.
        
        This function reads a dataset file (CSV, JSON, or Excel), sends each row's
        prompt to the LLaMA model, and saves the results.
        
        Args:
            dataset_path (str): Path to the input dataset file (.csv, .json, or .xlsx).
            prompt_column (str): Name of the column containing the prompts/questions.
            output_path (str): Path where the results will be saved.
            system_prompt (str, optional): System prompt to use for all queries.
            batch_size (int): Number of rows to process before showing progress.
        
        Returns:
            pd.DataFrame: DataFrame with original data plus 'llama_response' column.
        
        Example:
            >>> processor = LlamaInferenceProcessor()
            >>> results = processor.process_dataset(
            ...     dataset_path="patient_questions.csv",
            ...     prompt_column="question",
            ...     output_path="results.csv"
            ... )
        """
        print(f"\n{'='*60}")
        print(f"Starting Dataset Processing")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_path}")
        print(f"Prompt Column: {prompt_column}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")
        
        # Load the dataset based on file extension
        file_extension = os.path.splitext(dataset_path)[1].lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(dataset_path)
        elif file_extension == '.json':
            df = pd.read_json(dataset_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Use .csv, .json, or .xlsx")
        
        # Verify the prompt column exists
        if prompt_column not in df.columns:
            raise ValueError(f"Column '{prompt_column}' not found in dataset. Available columns: {list(df.columns)}")
        
        print(f"✓ Loaded dataset with {len(df)} rows")
        
        # Create a new column for responses
        responses = []
        
        # Process each row
        for idx, row in df.iterrows():
            prompt = str(row[prompt_column])
            
            # Show progress
            if (idx + 1) % batch_size == 0:
                print(f"Processing row {idx + 1}/{len(df)}...")
            
            # Get model response
            response = self.query_single(
                user_prompt=prompt,
                system_prompt=system_prompt
            )
            
            responses.append(response)
        
        # Add responses to dataframe
        df['llama_response'] = responses
        
        # Save results
        output_extension = os.path.splitext(output_path)[1].lower()
        if output_extension == '.csv':
            df.to_csv(output_path, index=False)
        elif output_extension == '.json':
            df.to_json(output_path, orient='records', indent=2)
        elif output_extension in ['.xlsx', '.xls']:
            df.to_excel(output_path, index=False)
        else:
            # Default to CSV if unknown extension
            df.to_csv(output_path, index=False)
        
        print(f"\n✓ Processing complete! Results saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return df
    
    
    def process_list(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Process a list of prompts and return results.
        
        Args:
            prompts (List[str]): List of prompts/questions to process.
            system_prompt (str, optional): System prompt for all queries.
        
        Returns:
            List[Dict[str, str]]: List of dictionaries with 'prompt' and 'response' keys.
        
        Example:
            >>> processor = LlamaInferenceProcessor()
            >>> prompts = ["What is hypertension?", "What causes diabetes?"]
            >>> results = processor.process_list(prompts)
        """
        results = []
        
        for idx, prompt in enumerate(prompts):
            print(f"Processing prompt {idx + 1}/{len(prompts)}...")
            
            response = self.query_single(
                user_prompt=prompt,
                system_prompt=system_prompt
            )
            
            results.append({
                'prompt': prompt,
                'response': response
            })
        
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """
    Main function demonstrating how to use the LlamaInferenceProcessor.
    
    INSTRUCTIONS FOR FUTURE USERS:
    1. Set your HF_TOKEN environment variable before running:
       export HF_TOKEN="your_huggingface_token_here"
    
    2. Uncomment the example that matches your use case below.
    
    3. Modify the parameters to match your dataset structure.
    """
    
    # Initialize the processor
    # Option 1: Use environment variable HF_TOKEN
    processor = LlamaInferenceProcessor()
    
    # Option 2: Pass API key directly (not recommended for production)
    # processor = LlamaInferenceProcessor(api_key="your_token_here")
    
    
    # ========================================================================
    # EXAMPLE 1: Single Query ( this will be UI interaction based )
    # ========================================================================
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Query")
    print("="*60 + "\n")
    
    
    # With system prompt (useful for clinical/medical context)
    response = processor.query_single(
        user_prompt="What are the symptoms of diabetes?",
        system_prompt="You are a helpful medical assistant. Provide accurate, evidence-based information."
    )
    print(f"Response: {response}\n")
    
    
    # ========================================================================
    # EXAMPLE 2: Process List of Prompts
    # ========================================================================
    print("\n" + "="*60)
    print("EXAMPLE 2: Process Multiple Prompts")
    print("="*60 + "\n")
    
    prompts = [
        "What is hypertension?",
        "What causes type 2 diabetes?",
        "What are the risk factors for heart disease?"
    ]
    
    results = processor.process_list(
        prompts=prompts,
        system_prompt="You are a medical AI assistant. Provide concise, accurate information."
    )
    
    for result in results:
        print(f"Q: {result['prompt']}")
        print(f"A: {result['response']}\n")
    
    
    # ========================================================================
    # EXAMPLE 3: Process Dataset from File
    # ========================================================================
    # UNCOMMENT THIS SECTION WHEN YOU HAVE YOUR DATASET READY
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Process Dataset from File")
    print("="*60 + "\n")
    
    # Example for CSV file with a 'question' column
    results_df = processor.process_dataset(
        dataset_path="your_dataset.csv",           # Path to your dataset
        prompt_column="question",                   # Column name with prompts
        output_path="results_with_responses.csv",  # Where to save results
        system_prompt="You are a clinical decision support AI assistant.",
        batch_size=10                               # Show progress every 10 rows
    )
    
    # Display first few results
    print(results_df.head())
    """
    
    print("\n✓ All examples completed successfully!")


if __name__ == "__main__":
    
    main()

