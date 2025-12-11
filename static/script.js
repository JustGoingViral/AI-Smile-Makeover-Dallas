// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorText = document.getElementById('errorText');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const retryBtn = document.getElementById('retryBtn');

let selectedFile = null;

// Initialize
function init() {
    setupEventListeners();
}

// Event Listeners
function setupEventListeners() {
    // Browse button
    browseBtn.addEventListener('click', () => fileInput.click());
    
    // Upload area click
    uploadArea.addEventListener('click', (e) => {
        if (e.target !== browseBtn) {
            fileInput.click();
        }
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Clear button
    clearBtn.addEventListener('click', resetUpload);

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);

    // New analysis button
    newAnalysisBtn.addEventListener('click', resetUpload);

    // Retry button
    retryBtn.addEventListener('click', resetToUpload);
}

// File Selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

// File Validation and Preview
function validateAndPreviewFile(file) {
    // Check file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPG or PNG).');
        return;
    }

    // Check file size (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size must be less than 10MB.');
        return;
    }

    selectedFile = file;
    displayPreview(file);
}

function displayPreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
        hideError();
    };
    
    reader.readAsDataURL(file);
}

// Analyze Image
async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }

    // Show loading
    previewSection.style.display = 'none';
    loadingSection.style.display = 'block';
    hideError();

    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Analysis failed. Please try again.');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Analysis error:', error);
        loadingSection.style.display = 'none';
        showError(error.message || 'Failed to analyze image. Please try again.');
    }
}

// Display Results
function displayResults(data) {
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // Animate score with circular progress
    animateScore(data.score);
    animateScoreCircle(data.score);

    // Display metrics
    document.getElementById('alignmentValue').textContent = data.alignment.toFixed(1);
    document.getElementById('symmetryValue').textContent = data.symmetry.toFixed(1);
    document.getElementById('colorValue').textContent = data.color_vitality.toFixed(1);

    // Animate metric bars
    setTimeout(() => {
        document.getElementById('alignmentBar').style.width = `${(data.alignment / 10) * 100}%`;
        document.getElementById('symmetryBar').style.width = `${(data.symmetry / 10) * 100}%`;
        document.getElementById('colorBar').style.width = `${(data.color_vitality / 10) * 100}%`;
    }, 100);

    // Display insight
    document.getElementById('insightText').textContent = data.insight;
}

// Animate Score Circle
function animateScoreCircle(targetScore) {
    const circle = document.getElementById('scoreProgress');
    if (!circle) return;
    
    const circumference = 565.48; // 2 * PI * 90
    const progress = (targetScore / 10) * circumference;
    const dashoffset = circumference - progress;
    
    setTimeout(() => {
        circle.style.strokeDashoffset = dashoffset;
    }, 200);
}

// Animate Score
function animateScore(targetScore) {
    const scoreElement = document.getElementById('scoreValue');
    const duration = 1500; // 1.5 seconds
    const steps = 60;
    const increment = targetScore / steps;
    let currentScore = 0;

    const interval = setInterval(() => {
        currentScore += increment;
        if (currentScore >= targetScore) {
            currentScore = targetScore;
            clearInterval(interval);
        }
        scoreElement.textContent = currentScore.toFixed(1);
    }, duration / steps);
}

// Reset Functions
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    hideError();
    
    // Reset metric bars
    document.getElementById('alignmentBar').style.width = '0';
    document.getElementById('symmetryBar').style.width = '0';
    document.getElementById('colorBar').style.width = '0';
}

function resetToUpload() {
    hideError();
    if (selectedFile) {
        previewSection.style.display = 'block';
    } else {
        uploadArea.style.display = 'block';
    }
}

// Error Handling
function showError(message) {
    errorText.textContent = message;
    errorSection.style.display = 'block';
    
    // Hide other sections
    uploadArea.style.display = 'none';
    previewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

function hideError() {
    errorSection.style.display = 'none';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);
