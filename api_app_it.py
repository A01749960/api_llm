from __future__ import annotations

import json
import random
import re
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

SEED = 7
random.seed(SEED)
rng = np.random.default_rng(SEED)

app = FastAPI(title="Topic Classifier (Italian)")

class TextPayload(BaseModel):
    text: str


def wikipedia_plaintext(title: str, *, user_agent: str = "mathematica-replica") -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }
    url = "https://it.wikipedia.org/w/api.php?" + urlencode(params)
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


try:
    import nltk
    from nltk.tokenize import sent_tokenize

    def _regex_split(text: str) -> List[str]:
        chunks = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in chunks if s.strip()]

    def split_sentences(text: str) -> List[str]:
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                return _regex_split(text)
            try:
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
            except Exception:
                return _regex_split(text)

except Exception:

    def split_sentences(text: str) -> List[str]:
        chunks = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in chunks if s.strip()]


def text_sentences(title: str) -> List[str]:
    return split_sentences(wikipedia_plaintext(title))


def build_model():
    physics = text_sentences("Fisica")
    biology = text_sentences("Biologia")
    math = text_sentences("Matematica")

    topicdataset = (
        [(s, "Fisica") for s in physics]
        + [(s, "Biologia") for s in biology]
        + [(s, "Matematica") for s in math]
    )

    X_train, y_train = zip(*topicdataset)
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2),
        MultinomialNB(),
    )
    model.fit(X_train, y_train)
    return model


model = None

@app.on_event("startup")
def _startup():
    global model
    model = build_model()


def topic(text: str) -> str:
    return model.predict([text])[0]


def topic_probabilities(text: str) -> Dict[str, float]:
    probs = model.predict_proba([text])[0]
    return dict(zip(model.classes_, probs))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify")
def classify(payload: TextPayload):
    return {
        "class": topic(payload.text),
        "probabilities": topic_probabilities(payload.text),
    }
