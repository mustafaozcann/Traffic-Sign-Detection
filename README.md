# Traffic Sign Detection

This project implements a deep learning-based traffic sign detection system using TensorFlow and Flask. The system can recognize 58 different types of traffic signs with high accuracy.

## Features

- Deep learning model trained on a comprehensive traffic sign dataset
- Web interface for easy image upload and prediction
- Real-time traffic sign classification
- Confidence score for predictions
- Support for 58 different traffic sign classes

## Technologies Used

- TensorFlow for model training and predictions
- Flask for web application
- OpenCV for image processing
- Python 3.x

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mustafaozcann/Traffic-Sign-Detection.git
cd Traffic-Sign-Detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`
3. Upload an image of a traffic sign
4. View the prediction results and confidence score

## Model Architecture

The model uses a CNN architecture with:
- Multiple convolutional layers with batch normalization
- Dropout layers for regularization
- Dense layers for classification
- Input image size: 32x32 pixels
- Output: 58 classes of traffic signs

## Performance

- Training accuracy: ~99%
- Validation accuracy: ~98.8%
- Early stopping implemented to prevent overfitting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- Mustafa Özcan
