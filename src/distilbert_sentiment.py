from transformers import pipeline


def load_distilbert():

    classifier = pipeline(

        "sentiment-analysis",

        model="distilbert-base-uncased-finetuned-sst-2-english"

    )

    return classifier


def predict_distilbert(text):

    classifier = load_distilbert()

    result = classifier(text)

    return result