# Persian News Classification

This repository contains Jupyter notebooks for analyzing and modeling a dataset of Persian news articles for classification purposes.

## Notebooks

- **EDA.ipynb**:  
  - Generates word clouds  
  - Plots the distribution of labels  
  - Displays dataframe information  

- **MachineLearning.ipynb**:  
  - Preprocesses the data  
  - Applies TF-IDF transformation  
  - Trains multiple machine learning models using grid search for hyperparameter tuning  
  - Models include ensemble methods, linear models, naive Bayes, nearest neighbors, decision trees, and gradient boosting  

- **DeepLearning.ipynb**:  
  - Trains Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN)  

For more details on the implementation and results, please refer to the individual notebooks.

## Python scripts

The notebooks are also available as a reusable package (`persian_news/`) with four entry points.

```
persian_news/
├── config.py          # all hyperparameters and paths (dataclasses)
├── preprocessing.py   # Persian cleaning, tokenization, lemmatization, stopwords
├── data.py            # CSV loading, preprocessing cache, stratified train/val/test split
├── metrics.py         # accuracy, precision, recall, F1, Cohen's kappa
├── ml.py              # TF-IDF + 14 classifiers tuned with GridSearchCV
├── dl/                # Word2Vec embeddings, TextCNN, LSTM, training loop with early stopping
├── results.py         # leaderboard and best-model selection
└── artifacts.py       # save / load the best model
train.py               # train everything, save the best model
predict.py             # classify new titles with the saved model
api.py                 # REST API + serves the built web page
frontend/              # the web page (React + Vite)
eda.py                 # label distribution + word cloud figures
```

```bash
# hazm needs Python 3.12 or 3.13 (not 3.14).
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put the dataset at data/farsnews_fainal.csv and stopword .txt files in data/persian_stopwords/
# (or pass --data / --stopwords-dir).
python train.py                                  # all ML + DL models (slow on a laptop)
python train.py --skip-ml --dl-models lstm       # only the LSTM
python train.py --tfidf-max-features 0           # ML with the full TF-IDF vocabulary
python predict.py "پرسپولیس با برتری مقابل استقلال قهرمان شد"
python eda.py --font data/Gandom.ttf
```

On a weak machine, train only the fast linear models. They also get a much larger TF-IDF vocabulary
than the default of 100 features, which is kept small so the full model zoo stays tractable:

```bash
python train.py --skip-dl --ml-models LogisticRegression SGDClassifier MultinomialNB \
    --tfidf-max-features 50000 --tfidf-max-ngram 2 --cv-folds 3 --n-jobs 2
```

To train on another machine (e.g. Colab), copy its `artifacts/best_model/` folder here afterwards, using
the same scikit-learn version.

### API and web page

The page needs Node.js 20.19+ or 22.12+ and one build; `api.py` serves the result from `frontend/dist/`.

```bash
cd frontend && npm install && npm run build && cd ..
python api.py        # then open http://127.0.0.1:8000  (API docs: /docs)
```

The server starts without a trained model too; the page then shows how to train one. Restart it after
training to load the new model, and after the first frontend build.

To work on the page with hot reload, run `python api.py` and, next to it, `npm run dev` in `frontend/`
(http://localhost:5173, which forwards `/api` to port 8000). If that port is taken, use another one for both:
`python api.py --port 8080` and `API_URL=http://127.0.0.1:8080 npm run dev`.

| Endpoint | Purpose |
|---|---|
| `GET /` | web page |
| `GET /api/model` | whether a model is loaded, its name, labels and test metrics |
| `POST /api/predict` | classify `{"text": "..."}`; returns `label`, `score`, `is_probability` and `scores` per label |

`/api/predict` answers 503 when no model is loaded and 422 when the text has no usable Persian words.
`is_probability` is false for models that only output margins (e.g. Perceptron), whose scores are only
comparable between labels.

All models are compared on the same validation split with one metric (`--metric`, Cohen's kappa
by default). The winner is evaluated once on the test split and saved to `artifacts/best_model/`
(`metadata.json` has its parameters and val/test metrics). The full leaderboard is written to
`artifacts/results.csv`.