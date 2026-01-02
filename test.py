from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def test_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        print("Loading model...")
        model = T5ForConditionalGeneration.from_pretrained("kompas_summarization_model_pt").to(device)
        tokenizer = T5Tokenizer.from_pretrained("kompas_summarization_tokenizer")
        
        print("Model loaded successfully! Testing with sample text...")
        text = "Contoh teks berita dalam bahasa Indonesia"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        print("Generating summary...")
        summary_ids = model.generate(inputs["input_ids"], max_length=50)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        print(f"Original: {text}")
        print(f"Summary: {summary}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    test_model()