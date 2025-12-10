"""
Flask application for AI Smile Makeover Dallas
Provides web interface and API for smile analysis
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import traceback

from utils.image_analysis import SmileAnalyzer, load_image_from_bytes, validate_image
from model.smile_scorer import create_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize analyzer and scorer
smile_analyzer = SmileAnalyzer()
smile_scorer = create_scorer()


def allowed_file(filename: str) -> bool:
    """
    Check if file has an allowed extension
    
    Args:
        filename: Name of the file
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """
    Render the main page with upload UI
    
    Returns:
        Rendered HTML template
    """
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        return "An error occurred loading the page", 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze uploaded smile image
    
    Returns:
        JSON response with analysis results or error message
    """
    try:
        # Validate request has file
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Validate file was selected
        if file.filename == '':
            logger.warning("Empty filename in request")
            return jsonify({'error': 'No image file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload JPG or PNG'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Validate file is not empty
        if len(image_bytes) == 0:
            logger.warning("Empty file uploaded")
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        logger.info(f"Processing image: {secure_filename(file.filename)} ({len(image_bytes)} bytes)")
        
        # Load image
        try:
            image_array = load_image_from_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return jsonify({'error': 'Failed to load image. Please ensure it is a valid image file'}), 400
        
        # Validate image
        is_valid, error_message = validate_image(image_array)
        if not is_valid:
            logger.warning(f"Image validation failed: {error_message}")
            return jsonify({'error': error_message}), 400
        
        # Analyze image for metrics
        try:
            metrics = smile_analyzer.analyze_image(image_array)
            logger.info(f"Metrics calculated: {metrics}")
        except ValueError as e:
            logger.error(f"Analysis failed: {e}")
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Failed to analyze image. Please try with a clearer face photo'}), 500
        
        # Generate score and insight
        try:
            results = smile_scorer.analyze(metrics)
            logger.info(f"Analysis complete: Score={results['score']}")
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Failed to generate score'}), 500
        
        # Return results
        return jsonify(results), 200
        
    except RequestEntityTooLarge:
        logger.warning("File too large")
        return jsonify({'error': 'File is too large. Maximum size is 10MB'}), 413
    except Exception as e:
        logger.error(f"Unexpected error in analyze endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred. Please try again'}), 500


@app.route('/health')
def health():
    """
    Health check endpoint
    
    Returns:
        JSON response with health status
    """
    return jsonify({
        'status': 'healthy',
        'service': 'AI Smile Makeover Dallas'
    }), 200


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file too large errors"""
    return jsonify({'error': 'File is too large. Maximum size is 10MB'}), 413


if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting AI Smile Makeover Dallas on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )
