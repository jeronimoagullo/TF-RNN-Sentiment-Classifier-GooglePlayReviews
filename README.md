# TensorFlow Sentiment Analysis on Google Play Reviews using Recurrent Neural Network (RNN)

## Project Description
This project trains a Recurrent Neural Network (RNN) to analyze user reviews from the Google Play Store and classify them as **positive**, **neutral**, or **negative**. Leveraging deep learning techniques, this project aims to provide insights into user sentiment, which can inform app development and marketing strategies.

## **Technologies and Libraries**
This project leverages several essential Python libraries:
- **TensorFlow & Keras**: For constructing, training, and evaluating the RNN model.
- **Pandas**: For data manipulation and analysis.
- **scikit-learn**: For dataset splitting and metric evaluation.
- **Matplotlib**: For data visualization.

## Table of Contents
- [TensorFlow Sentiment Analysis on Google Play Reviews using Recurrent Neural Network (RNN)](#tensorflow-sentiment-analysis-on-google-play-reviews-using-recurrent-neural-network-rnn)
  - [Project Description](#project-description)
  - [**Technologies and Libraries**](#technologies-and-libraries)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

---

## Installation
To run this project, you need Python 3 and the following dependencies:
- `tensorflow`
- `keras`
- `pandas`
- `numpy`
- `sklearn`
- `matplotlib`
- `pathlib`
- `pickle` (if you want to save tokenizer)

Install these dependencies with:
```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib pathlib pickle
```

## Dataset
The project uses a dataset of Google Play Store user reviews, `googleplaystore_user_reviews.csv`, with a total of 64,295 entries. Only records with non-null reviews and sentiment labels are used for training, which results in around 37,400 labeled entries. The dataset includes:
- **Review**: Text of the review.
- **Sentiment**: Label indicating if the review is positive, neutral, or negative.

## Model Architecture
The model is built using a Recurrent Neural Network (RNN) architecture, ideal for processing sequential data like text. Key layers and components include:
- **Embedding Layer**: For converting text into dense vector representations.
- **LSTM/GRU Layers**: To capture contextual information from the sequence of words in the review.
- **Dense Output Layer**: A fully connected layer with three neurons (for positive, neutral, and negative classes), activated by softmax.

## Training
To train the model, the dataset is split into training and testing sets. During training:
- Text preprocessing (tokenization and padding) is applied to standardize input sequences.
- The model is optimized using categorical cross-entropy as the loss function and the Adam optimizer.
- Training metrics are monitored to ensure the model's accuracy and reduce overfitting.

You can checkout the training process by running the Jupyter notebook cells sequentially.

## Evaluation
After training, the model is evaluated on the test set, with metrics like accuracy and F1 score used to measure its performance. Additionally, a confusion matrix may be included to visualize the prediction distribution.

## Results
The model demonstrates a strong ability to classify sentiments accurately:
- **Accuracy**: The model achieves satisfactory accuracy on both training and test datasets.
- **Visualizations**: Training and validation loss/accuracy plots provide insights into the model's performance and convergence.

## Usage
The notebook includes code to predict sentiment for new reviews, replacing the variable `sample_review` with any review text to classify its sentiment:

```python
predicted_sentiment = model.predict(sample_review)
print(predicted_sentiment)
```

## Contributing
Contributions are welcome! If you'd like to improve this project or extend its functionality, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
