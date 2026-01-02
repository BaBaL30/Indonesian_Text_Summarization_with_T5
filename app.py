from flask import Flask, request, jsonify, render_template
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def load_model():
    try:
        model_path = os.path.join("kompas_summarization_model_pt")
        tokenizer_path = os.path.join("kompas_summarization_tokenizer")
        
        print(f"MODEL PATH: {model_path}")
        print(f"TOKENIZER PATH: {tokenizer_path}")
        
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        
        print("✅ Model dan tokenizer berhasil dimuat")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Gagal load model: {str(e)}")
        return None, None


# Load model at startup
model, tokenizer = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    logger.info("Received summarize request")
    
    # Check if model is loaded
    if model is None or tokenizer is None:
        error_msg = "Model not loaded properly"
        logger.error(error_msg)
        return jsonify({
            "error": "Model Error",
            "message": error_msg,
            "status": 500
        }), 500

    # Check content type
    if not request.is_json:
        error_msg = "Content-Type must be application/json"
        logger.error(error_msg)
        return jsonify({
            "error": "Invalid Content-Type",
            "message": error_msg,
            "status": 415
        }), 415

    try:
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        
        if not data or 'text' not in data:
            error_msg = "Missing 'text' in request body"
            logger.error(error_msg)
            return jsonify({
                "error": "Invalid Input",
                "message": error_msg,
                "status": 400
            }), 400

        text = data['text'].strip()
        if not text:
            error_msg = "Text cannot be empty"
            logger.error(error_msg)
            return jsonify({
                "error": "Empty Text",
                "message": error_msg,
                "status": 400
            }), 400

        logger.info(f"Processing text (length: {len(text)} characters)")
        
        # Tokenize and generate summary
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        logger.info("Generating summary...")
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=256,
            min_length=64,
            num_beams=2,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.info(f"Generated summary (length: {len(summary)} characters)")

        return jsonify({
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })

    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "error": "Processing Error",
            "message": error_msg,
            "status": 500
        }), 500

if __name__ == '__main__':
    # Ensure model directories exist
    os.makedirs("kompas_summarization_model_pt", exist_ok=True)
    os.makedirs("kompas_summarization_tokenizer", exist_ok=True)
    
    try:
        logger.info("Starting Flask application")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")