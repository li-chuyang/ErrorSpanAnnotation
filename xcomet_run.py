import os
from comet import download_model, load_from_checkpoint


def run_xcomet_inference():
    model_name = "Unbabel/XCOMET-XL"
    
    # Locate model checkpoint using system environment variables
    print(f"Action: Locating model checkpoint for {model_name}")
    checkpoint_path = download_model(model_name)
    print(f"Status: Checkpoint path identified at {checkpoint_path}")

    # Security check to ensure no data is written to the home directory
    if "/cl/home2/share" not in checkpoint_path:
        print("Critical Error: Model path is not pointing to the shared directory.")
        return

    # Load model to GPU
    print("Action: Loading model from checkpoint to GPU")
    model = load_from_checkpoint(checkpoint_path)

    # Test data
    data = [
        {
            "src": "The basic principle of the project is to ensure safety.",
            "mt": "这个项目的原则是基本确保安全的。",
            "ref": "这个项目基本的原则是确保安全。"
        },
        {
            "src": "The food was very delicious, but the service was slow.",
            "mt": "这个食物非常美味，但服务很慢。",
            "ref": "这个食物很美味，但服务很慢。"
        }
    ]

    # Model inference
    print("Action: Starting model inference")
    outputs = model.predict(data, batch_size=2, gpus=1)

    print("\n" + "="*60)
    print("XCOMET EVALUATION REPORT")
    print("="*60)

    for i, (score, spans) in enumerate(zip(outputs.scores, outputs.metadata.error_spans)):
        print(f"\n[Sample {i+1}]")
        print(f"Overall Quality Score: {score:.4f}")
        
        if spans:
            for span in spans:
                print(f"  - Severity: {span['severity']} | Text: '{span['text']}' | Range: {span['start']}-{span['end']}")
        else:
            print("No error spans detected.")

    print("\n" + "="*60)
    print(f"Global System Score: {outputs.system_score:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_xcomet_inference()