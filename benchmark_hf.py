import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def measure_tps(
    model_id: str, 
    prompt: str, 
    max_new_tokens: int = 128, 
    device: str = "cuda", 
    dtype: torch.dtype = torch.bfloat16
):
    print(f"Loading standard Hugging Face model from: {model_id}")
    print(f"Device: {device} | Dtype: {dtype}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Standard HF Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    ).eval()

    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    # --- WARM-UP ---
    # The first forward pass on a GPU includes lazy initialization of CUDA contexts 
    # and kernels. We must run a throwaway generation first so it doesn't skew the timer.
    print("Running warm-up step...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)

    # --- BENCHMARK ---
    print(f"Running benchmark (target: {max_new_tokens} tokens)...")
    
    # Synchronize ensures the GPU has finished all previous tasks before we start the clock
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        # Using min_new_tokens ensures the model doesn't stop early by emitting an EOS token,
        # giving us a consistent workload to measure.
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            min_new_tokens=max_new_tokens,
            do_sample=False, # Force greedy decoding for consistent timing
            use_cache=True   # Standard HuggingFace KV cache
        )
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    # --- CALCULATE ---
    # Model output includes the prompt, so we subtract the input length
    generated_tokens = outputs.shape[1] - input_length
    elapsed_time = end_time - start_time
    tps = generated_tokens / elapsed_time
    
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    print("\n" + "="*40)
    print(" BASELINE RESULTS")
    print("="*40)
    print(f"Generated text:  {generated_text.strip()}...")
    print(f"Tokens decoded:  {generated_tokens}")
    print(f"Elapsed time:    {elapsed_time:.3f} seconds")
    print(f"Throughput:      **{tps:.2f} tokens/second**")
    print("="*40)

if __name__ == "__main__":
    # You can point this to the official Hugging Face hub ID (e.g., "Qwen/Qwen3-8B")
    # or the local directory where you downloaded the official weights.
    MODEL_ID = "Qwen/Qwen3-1.7B" 
    TEST_PROMPT = "The history of quantum computing is"
    
    measure_tps(MODEL_ID, TEST_PROMPT)