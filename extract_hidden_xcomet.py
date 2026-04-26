import torch
import pandas as pd
from tqdm import tqdm
from comet import download_model, load_from_checkpoint
from ESA.annotation_loader import AnnotationLoader

# --- 1. Data Preparation ---
print("Loading ESA dataset...")
loader = AnnotationLoader(refresh_cache=False)
df = loader.get_view(["ESA-1"], only_overlap=False)
error_df = df[df['ESA-1_error_spans'].notna()].copy()

def has_valid_spans(spans):
    if not isinstance(spans, list): return False
    return any(str(s.get('start_i', 'missing')).isdigit() for s in spans)

valid_error_df = error_df[error_df['ESA-1_error_spans'].apply(has_valid_spans)]
print(f"Total valid high-quality samples: {len(valid_error_df)}")

# --- 2. Load Model ---
print("Loading xCOMET model...")
model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)
model.eval()
if torch.cuda.is_available(): model.to("cuda")
tokenizer = model.encoder.tokenizer

# --- 3. Alignment Logic ---
def get_token_labels(error_spans, offsets):
    token_labels = [0] * len(offsets)
    for span in error_spans:
        s, e = span.get('start_i'), span.get('end_i')
        if not str(s).isdigit() or not str(e).isdigit(): continue
        s, e = int(s), int(e)
        for i, (tok_s, tok_e) in enumerate(offsets):
            if tok_s == tok_e: continue
            if max(s, tok_s) < min(e, tok_e):
                token_labels[i] = 1
    return token_labels

# --- 4. Extraction with Binary Saving ---
all_vectors = []
all_labels = []
all_tokens =[]

# Process all valid samples
target_samples = valid_error_df 

for idx, row in tqdm(target_samples.iterrows(), total=len(target_samples), desc="Extracting Neurons"):
    source, mt, spans = row['source'], row['hypothesis'], row['ESA-1_error_spans']
    
    # Step A: Tokenize ONLY the MT to get our labels and pure MT tokens
    enc = tokenizer(mt, return_offsets_mapping=True, add_special_tokens=True)
    labels = get_token_labels(spans, enc['offset_mapping'])
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])
    
    # Step B: Prepare input for xCOMET (Source + MT combined)
    input_data =[{"src": source, "mt": mt, "ref": "", "score": 0.0}]
    prepared = model.prepare_sample(input_data)
    
    # Unwrap the nested tuple/list to get the dictionary
    model_input = prepared
    while isinstance(model_input, (tuple, list)): 
        model_input = model_input[0]
    
    if torch.cuda.is_available():
        model_input = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in model_input.items()}

    # Step C: Extract Hidden States from the model
    with torch.no_grad():
        # Pass ONLY the required parameters to the underlying XLM-RoBERTa
        outputs = model.encoder.model(
            input_ids=model_input["input_ids"], 
            attention_mask=model_input["attention_mask"], 
            output_hidden_states=True
        )
        # Full hidden states including BOTH Source and MT. Shape: [Full_Seq_Len, 2560]
        full_hidden_states = outputs.hidden_states[-1][0]   
         
    # Step D: The crucial "Alignment Fix" (Find where MT starts in the full sequence)
    # We strip the first and last tokens (usually <s> and </s>) from MT to search for the core sequence
    core_mt_ids = enc['input_ids'][1:-1]
    full_ids = model_input["input_ids"][0].cpu().tolist()
    
    # Search for the core MT sequence inside the full combined sequence
    start_idx = -1
    window_size = len(core_mt_ids)
    for j in range(len(full_ids) - window_size + 1):
        if full_ids[j : j + window_size] == core_mt_ids:
            start_idx = j
            break
            
    if start_idx != -1:
        # Match found! We slice the hidden states, tokens, and labels to keep ONLY the core MT part
        # This completely drops the source language neurons and the special tokens
        mt_hidden_states = full_hidden_states[start_idx : start_idx + window_size]
        mt_labels = labels[1:-1]
        mt_tokens = tokens[1:-1]
        
        # Save to our global lists
        for i in range(window_size):
            all_vectors.append(mt_hidden_states[i].cpu()) 
            all_labels.append(mt_labels[i])
            all_tokens.append(mt_tokens[i])
    else:
        # If alignment fails due to extreme tokenization differences (very rare)
        print(f"\nWarning: Could not align MT sequence for sample {idx}. Skipping.")

# --- 5. Merge and Save ---
print("\nStacking tensors...")
final_data = {
    "X": torch.stack(all_vectors), 
    "y": torch.tensor(all_labels),
    "tokens": all_tokens
}

# Save as binary file
torch.save(final_data, "experiment_data.pt")
print(f"Successfully saved {len(all_labels)} aligned tokens to experiment_data.pt")