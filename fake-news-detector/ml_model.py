"""ML model for fake news detection using TF-IDF + Logistic Regression."""

import os
import csv
import random
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'news.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'trained_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')


def generate_dataset():
    """Generate a synthetic news dataset with ~1000 labeled headlines."""
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    fake_templates = [
        "BREAKING: {person} caught {action} in secret {place}!!!",
        "SHOCKING: Government hiding {thing} from the public",
        "You WON'T BELIEVE what {person} did to {target}!!!",
        "EXPOSED: The truth about {thing} they don't want you to know",
        "ALERT: {thing} causes {effect} — doctors are STUNNED",
        "SECRET {thing} discovered that {action} overnight",
        "WARNING: {person} planning to {action} — leaked documents reveal",
        "CONFIRMED: {thing} is a complete HOAX created by {group}",
        "{person} EXPOSED for {action} — the evidence is UNDENIABLE",
        "URGENT: {thing} will {effect} by next week!!!",
        "Scientists BAFFLED by {thing} that {action}",
        "BOMBSHELL: {person} secretly {action} behind closed doors",
        "THEY DON'T WANT YOU TO KNOW: {thing} exposed",
        "Is {thing} secretly controlled by {group}? Evidence says YES",
        "MIRACLE {thing} that {action} — Big Pharma HATES this",
        "BREAKING NEWS: {person} admits to {action} on live TV!!!",
        "TOP SECRET: {group} caught planning {thing}",
        "LOOK what {person} was caught doing at {place}!!!",
        "PROOF: {thing} is being used to {action}",
        "CENSORED: The {thing} story mainstream media won't cover",
    ]

    real_templates = [
        "Federal Reserve announces {percent}% interest rate adjustment",
        "{company} reports quarterly earnings of ${amount} billion",
        "Senate passes bipartisan {topic} reform bill",
        "New study finds link between {thing} and {health_outcome}",
        "{city} mayor announces infrastructure development plan",
        "International summit addresses {topic} concerns",
        "Supreme Court to hear arguments on {topic} case",
        "{country} signs trade agreement with {partner}",
        "Annual report shows {percent}% change in {metric}",
        "University researchers publish findings on {topic}",
        "Department of {dept} releases updated guidelines",
        "{company} announces expansion into {region} market",
        "National weather service issues advisory for {region}",
        "Economic indicators suggest {trend} in consumer spending",
        "Healthcare officials recommend {action} for flu season",
        "Local school district approves new {topic} curriculum",
        "Transportation authority unveils {project} improvement plan",
        "Environmental agency reports progress on {topic} goals",
        "{company} to invest ${amount} million in renewable energy",
        "Census data reveals population trends in {region}",
    ]

    persons = ["the President", "a Senator", "a celebrity", "the CEO", "a billionaire",
               "a politician", "a tech mogul", "a Hollywood star", "a world leader", "an insider"]
    actions = ["manipulating elections", "hiding evidence", "stealing money",
               "lying to citizens", "destroying documents", "making secret deals",
               "spying on people", "covering up scandals", "faking data",
               "conspiring against democracy"]
    things = ["5G towers", "vaccines", "the economy", "social media", "elections",
              "climate data", "food supply", "water supply", "education system", "healthcare"]
    places = ["a bunker", "a private island", "the White House", "a mansion", "a lab"]
    targets = ["citizens", "children", "voters", "taxpayers", "small businesses"]
    effects = ["destroy the economy", "cause widespread panic", "change everything",
               "collapse society", "end democracy"]
    groups = ["the elite", "secret societies", "the deep state", "shadow government",
              "big corporations"]

    companies = ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Meta",
                 "Netflix", "Samsung", "Intel", "Boeing"]
    cities = ["New York", "Chicago", "Los Angeles", "Houston", "Phoenix",
              "Seattle", "Denver", "Atlanta", "Boston", "Portland"]
    countries = ["United States", "Canada", "United Kingdom", "Germany", "Japan",
                 "Australia", "France", "South Korea", "India", "Brazil"]
    topics = ["immigration", "education", "healthcare", "cybersecurity", "climate",
              "housing", "transportation", "technology", "energy", "agriculture"]
    health_outcomes = ["improved cardiovascular health", "reduced anxiety",
                       "better sleep quality", "lower blood pressure",
                       "increased cognitive function"]
    depts = ["Education", "Health", "Energy", "Commerce", "Transportation",
             "Labor", "Defense", "Agriculture", "Justice", "Interior"]
    regions = ["the Midwest", "the Northeast", "the Pacific Northwest",
               "the Southeast", "the Southwest", "Northern Europe", "East Asia"]
    trends = ["an increase", "a decrease", "stability", "a slight uptick",
              "a notable shift"]
    metrics = ["employment", "housing prices", "retail sales",
               "manufacturing output", "GDP growth"]
    projects = ["highway", "rail", "bridge", "airport", "public transit"]

    headlines = []

    # Generate ~500 fake headlines
    for _ in range(500):
        template = random.choice(fake_templates)
        headline = template.format(
            person=random.choice(persons),
            action=random.choice(actions),
            thing=random.choice(things),
            place=random.choice(places),
            target=random.choice(targets),
            effect=random.choice(effects),
            group=random.choice(groups),
        )
        headlines.append((headline, 'FAKE'))

    # Generate ~500 real headlines
    for _ in range(500):
        template = random.choice(real_templates)
        headline = template.format(
            company=random.choice(companies),
            city=random.choice(cities),
            country=random.choice(countries),
            partner=random.choice(countries),
            topic=random.choice(topics),
            thing=random.choice(["exercise", "diet", "meditation", "reading", "sleep"]),
            health_outcome=random.choice(health_outcomes),
            percent=random.randint(1, 15),
            amount=random.randint(1, 50),
            dept=random.choice(depts),
            region=random.choice(regions),
            trend=random.choice(trends),
            metric=random.choice(metrics),
            project=random.choice(projects),
            action="preventive vaccination",
        )
        headlines.append((headline, 'REAL'))

    random.shuffle(headlines)

    with open(DATASET_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['headline', 'label'])
        writer.writerows(headlines)

    print(f"Dataset generated: {len(headlines)} headlines saved to {DATASET_PATH}")
    return DATASET_PATH


def train_model():
    """Train the fake news detection model."""
    # Generate dataset if it doesn't exist
    if not os.path.exists(DATASET_PATH):
        generate_dataset()

    # Load dataset
    headlines = []
    labels = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            headlines.append(row['headline'])
            labels.append(row['label'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        headlines, labels, test_size=0.2, random_state=42
    )

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    accuracy = model.score(X_test_tfidf, y_test)
    print(f"Model trained — Accuracy: {accuracy:.2%}")

    # Save model and vectorizer
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_DIR}")

    return model, vectorizer


def load_model():
    """Load the trained model and vectorizer, training if needed."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return train_model()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(headline, model=None, vectorizer=None):
    """
    Predict whether a headline is fake or real.

    Returns:
        dict: {'label': 'FAKE'|'REAL', 'confidence': float (0-100)}
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    # Transform input
    X = vectorizer.transform([headline])

    # Get prediction and probability
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # Get confidence for the predicted class
    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index] * 100

    return {
        'label': prediction,
        'confidence': round(confidence, 1)
    }


if __name__ == '__main__':
    # Train model when run directly
    generate_dataset()
    model, vectorizer = train_model()

    # Test predictions
    test_headlines = [
        "BREAKING: Government hiding aliens from the public!!!",
        "Federal Reserve announces 3% interest rate adjustment",
        "You WON'T BELIEVE what the President did to citizens!!!",
        "University researchers publish findings on climate",
        "SHOCKING: Vaccines causes widespread panic — doctors are STUNNED",
    ]

    print("\n--- Test Predictions ---")
    for h in test_headlines:
        result = predict(h, model, vectorizer)
        print(f"  [{result['label']}] ({result['confidence']:.1f}%) {h[:60]}...")
