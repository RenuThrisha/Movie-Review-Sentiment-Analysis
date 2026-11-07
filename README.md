# Movie Review Sentiment Analysis

A small sentiment-analysis project that classifies a movie review as **positive** or **negative** using a SimpleRNN trained on the IMDB movie-review dataset. A Streamlit app provides an interface for entering a review and viewing the predicted sentiment and score.

Demo Link:https://movie-review-sentiment-analysis-bhyxlj6qu7cofbzrvjwhpz.streamlit.app/

## Project files

- `app.py` — Streamlit application; loads `rnn_imdb.h5`, preprocesses the entered review, and displays a prediction.
- `rnn_imdb.h5` — saved Keras model used by the app.
- `simplernn.ipynb` — loads and pads the IMDB dataset, defines and trains the SimpleRNN, evaluates it, and saves the model.
- `prediction.ipynb` — demonstrates loading the saved model and predicting sentiment.
- `requirements.txt` — Python package dependencies.

## Setup

Use Python 3.10 or later, then run these commands from the project directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

Make sure `rnn_imdb.h5` is in the project directory, then run:

```powershell
streamlit run app.py
```

Streamlit opens the app in a browser. Enter a movie review and select **Classify** to see the predicted sentiment and model score.

## Train the model

Open `simplernn.ipynb` in Jupyter or VS Code and run its cells in order. The notebook uses the IMDB dataset provided by TensorFlow/Keras, limits the vocabulary to 10,000 words, and pads reviews to 500 tokens. It evaluates the trained model on the test set and saves it as `rnn_imdb.h5` in the current working directory. Place that file in the project directory before starting the app.

## Important model note

The training notebook currently uses a ReLU activation in its final `Dense` layer with binary cross-entropy. The app interprets the model output as a probability and applies a 0.5 threshold. For binary classification, use a sigmoid output instead:

```python
Dense(1, activation="sigmoid")
```

After changing the output activation, retrain and save the model before using the app. Do not treat the displayed score as a calibrated probability unless the model has been validated for that purpose.

## Dependencies

Dependencies are listed in `requirements.txt`: TensorFlow, Streamlit, pandas, and NumPy.
