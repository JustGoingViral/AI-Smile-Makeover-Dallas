# AI Smile Makeover Dallas - Setup Guide

## Overview
AI Smile Makeover Dallas is a Flask-based web application that analyzes smile photos using advanced AI technology. It uses MediaPipe for facial landmark detection, OpenCV and NumPy for image analysis, and PyTorch for scoring.

## Features
- **Responsive Web UI**: Clean, modern interface with drag-and-drop image upload
- **Real-time Analysis**: Instant smile analysis with visual feedback
- **Comprehensive Metrics**:
  - Alignment: Evaluates horizontal balance and centering
  - Symmetry: Measures facial symmetry
  - Color Vitality: Analyzes brightness, saturation, and contrast
- **AI Scoring**: PyTorch-based model provides 1-10 overall score
- **Professional Insights**: Personalized recommendations based on analysis

## Requirements
- Python 3.8 or higher
- pip (Python package manager)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/JustGoingViral/AI-Smile-Makeover-Dallas.git
cd AI-Smile-Makeover-Dallas
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Development Mode
```bash
python app.py
```

The application will start on `http://localhost:5000`

### Production Mode
For production deployment, use a WSGI server like Gunicorn:

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Usage

1. **Open the application** in your web browser at `http://localhost:5000`
2. **Upload an image**:
   - Click "Browse Files" or drag and drop an image
   - Supported formats: JPG, PNG (max 10MB)
3. **Analyze**: Click the "Analyze Smile" button
4. **View Results**:
   - Overall score (1-10)
   - Individual metrics (Alignment, Symmetry, Color Vitality)
   - Professional insight and recommendations

## Project Structure
```
AI-Smile-Makeover-Dallas/
├── app.py                  # Flask application (main entry point)
├── requirements.txt        # Python dependencies
├── static/
│   ├── styles.css         # CSS styling
│   └── script.js          # JavaScript for UI interactions
├── templates/
│   └── index.html         # Main HTML template
├── utils/
│   ├── __init__.py
│   └── image_analysis.py  # MediaPipe & OpenCV analysis utilities
└── model/
    ├── __init__.py
    └── smile_scorer.py     # PyTorch scoring model
```

## API Endpoints

### GET /
Returns the main HTML page with upload UI

### POST /analyze
Analyzes an uploaded image and returns JSON results

**Request:**
- Content-Type: multipart/form-data
- Body: image file (key: "image")

**Response:**
```json
{
  "score": 8.5,
  "alignment": 8.2,
  "symmetry": 8.9,
  "color_vitality": 8.3,
  "insight": "Excellent smile! Your smile demonstrates..."
}
```

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "service": "AI Smile Makeover Dallas"
}
```

## Error Handling
The application includes comprehensive error handling:
- Invalid file types
- File size validation (max 10MB)
- Face detection failures
- Image processing errors
- All errors return appropriate HTTP status codes and user-friendly messages

## Technical Details

### MediaPipe Face Mesh
- Detects 468 facial landmarks
- Used for alignment and symmetry calculations
- High accuracy face detection

### Image Analysis
- **Alignment**: Evaluates mouth corner positions and centering
- **Symmetry**: Analyzes left/right facial balance
- **Color Vitality**: Examines brightness, saturation, contrast

### PyTorch Model
- Stub implementation for demonstration
- Can be replaced with trained neural network
- Combines metrics into overall 1-10 score

## Environment Variables
- `PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: False)

## Security
- File size limits enforced
- File type validation
- Secure file handling
- Input sanitization
- No persistent storage of uploaded images

## Browser Support
- Chrome (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers supported

## Troubleshooting

### Issue: "No face detected"
- Ensure the photo clearly shows the face
- Use good lighting
- Face should be front-facing

### Issue: Dependencies fail to install
- Ensure Python 3.8+ is installed
- Try upgrading pip: `pip install --upgrade pip`
- Install system dependencies if needed (OpenCV may require system libraries)

## License
© 2024 AI Smile Makeover Dallas. All rights reserved.

## Support
For issues or questions, please open an issue on the GitHub repository.
