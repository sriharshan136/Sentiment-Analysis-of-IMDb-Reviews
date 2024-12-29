# Sentiment Analysis of IMDb Movie Reviews

This repository contains a sentiment analysis project on the IMDb dataset of 50,000 movie reviews. The objective of this project is to classify movie reviews as **positive** or **negative** using advanced Natural Language Processing (NLP) techniques.

## Dataset

The dataset used in this project is the [IMDb 50K Movie Reviews Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle. It consists of 25,000 training samples and 25,000 test samples, evenly split between positive and negative reviews.

## Project Workflow

### 1. **Data Preprocessing**
- Removed HTML tags and special characters.
- Lowercased the text.
- Tokenized and padded sequences.
- Split the data into training and validation sets.

### 2. **Techniques Used**
- Word embeddings for feature representation.
- Deep learning models for classification.
  
### 3. **Modeling**
- **Technique:** [Specify the technique you used, e.g., LSTM, BERT, CNN, etc.]
- **Framework:** TensorFlow/Keras or PyTorch (as applicable).
- Evaluated the model using metrics such as accuracy, precision, recall, and F1-score.

### 4. **Results**
- The model achieved a validation accuracy of [Insert Accuracy] on the test set.
- [Include any key findings or observations.]

## Installation and Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/sriharshan136/Sentiment-Analysis-of-IMDb-Reviews.git
   cd Sentiment-Analysis-of-IMDb-Reviews
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Sentiment_Analysis.ipynb
   ```

4. Run the cells to execute the sentiment analysis pipeline.

## Requirements

- Python 3.7+
- TensorFlow/PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib/Seaborn

## Repository Structure

```
.
├── Sentiment_Analysis.ipynb  # Main notebook
├── requirements.txt          # List of required Python packages
├── README.md                 # Project documentation
└── data/                     # Dataset folder (not included in repo)
```

## Future Enhancements

- Experimenting with transformer-based models like BERT.
- Fine-tuning hyperparameters for better performance.
- Visualizing results with more advanced techniques.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for the IMDb dataset.
- [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) for deep learning frameworks.

Feel free to contribute or provide feedback by creating an issue or pull request.
