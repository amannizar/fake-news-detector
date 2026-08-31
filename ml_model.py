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
REAL_DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'fake_or_real_news.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'trained_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')


# ---------------------------------------------------------------------------
# Improved synthetic data generator — realistic patterns, not obvious ones
# ---------------------------------------------------------------------------

def generate_dataset():
    """Generate a synthetic news dataset with realistic labeled headlines."""
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    # --- FAKE templates: realistic-sounding misinformation ---
    # These mimic real fake news: opinion presented as fact, misleading framing,
    # conspiracy-adjacent claims, political spin — NOT all-caps clickbait.
    fake_templates = [
        # Conspiracy / secret knowledge
        "Exposed: The hidden agenda behind {topic} that no one is talking about",
        "Former {role} reveals the truth about {topic} they tried to silence",
        "Leaked documents show {group} has been covering up {topic} for years",
        "Insider confesses: {topic} was never what we were told",
        "What they are not telling you about {topic} will change everything",
        "Whistleblower exposes massive {topic} coverup at {org}",
        "Declassified files confirm {group} lied about {topic}",
        "Deep investigation reveals the real reason behind {topic}",

        # Misleading health / science claims
        "{study} study proves {food} causes {health_problem} in 90 percent of cases",
        "Doctors do not want you to know about this {treatment} for {health_problem}",
        "Natural remedy that {industry} does not want you to discover",
        "Shocking {study} findings on {topic} suppressed by {group}",
        "New research confirms what they have been hiding about {vaccine}",
        "{food} linked to {health_problem} in major new {study} study",
        "Scientists who spoke out about {topic} were silenced by {org}",
        "The {health_problem} epidemic they created and do not want to fix",

        # Political spin / misleading claims
        "{politician} secretly {action} while claiming to support {topic}",
        "Report: {group} spent millions to manipulate public opinion on {topic}",
        "Exposed: {politician} connection to {group} raises serious questions",
        "The {policy} that will destroy {industry} and nobody noticed",
        "Why {politician} is really pushing the {policy} agenda",
        "{group} caught funding propaganda about {topic}",
        "New {policy} quietly gives {group} unprecedented power over {topic}",
        "Former {role} warns {politician} is lying about {topic}",

        # Fake science / health misinformation
        "{food} now linked to {health_problem} according to suppressed study",
        "Leading {role} admits {vaccine} has been causing {health_problem} all along",
        "This {treatment} cures {health_problem} but {industry} does not want you to know",
        "Major {study} study reveals {topic} dangers the government ignored",
        "Hidden ingredients in {food} are causing {health_problem} epidemic",
        "{role} breaks silence on {topic} coverup",
        "The {policy} lie: how {group} is using {topic} to control the public",

        # Misleading financial / economic claims
        "Economists warn {policy} will cause {economic_problem} within months",
        "{group} is secretly {economic_action} your retirement savings",
        "Exposed: {industry} insider reveals {topic} is a complete scam",
        "Former {role} predicts {economic_problem} due to {policy}",
        "How {group} is using {topic} to steal from ordinary citizens",
    ]

    roles = ["government official", "intelligence officer", "FDA scientist",
             "military commander", "bank executive", "senior researcher",
             "hospital director", "intelligence analyst", "corporate executive",
             "congressional staffer"]
    groups = ["the government", "big pharma", "the media", "tech companies",
              "wall street", "the establishment", "global elites", "special interests"]
    orgs = ["the CDC", "the WHO", "the FDA", "major universities", "the government",
            "the Pentagon", "Silicon Valley", "mainstream media"]
    topics_health = ["vaccine safety", "processed food", "water fluoridation",
                     "air quality", "prescription drugs", "medical testing",
                     "genetic modification", "pesticide use"]
    health_problems = ["chronic illness", "autoimmune disorders", "cognitive decline",
                       "heart disease", "developmental issues", "hormone disruption",
                       "immune system damage", "neurological problems"]
    foods = ["processed foods", "artificial sweeteners", "genetically modified crops",
             "fluoridated water", "pasteurized milk", "microplastics in seafood"]
    treatments = ["treatment", "supplement", "therapy", "natural approach",
                  "breakthrough remedy", "alternative medicine"]
    industries = ["pharmaceutical", "food", "medical", "agricultural", "tech"]
    politicians = ["Senator", "Governor", "the President", "the Secretary",
                   "a Congressman", "the Chancellor"]
    policies = ["new healthcare", "tax reform", "education", "climate",
                "immigration", "regulatory", "surveillance"]
    economic_problems = ["a market crash", "severe inflation", "mass unemployment",
                         "a debt crisis", "a housing collapse"]
    economic_actions = ["gambling with", "devaluing", "secretly转移ing",
                        "manipulating", "liquidating"]
    vaccines = ["the new vaccine", "recent vaccines", "childhood vaccines",
                "flu vaccine", "COVID vaccine", "mRNA vaccines"]
    studies = ["Harvard", "Stanford", "Oxford", "Johns Hopkins", "Yale",
               "peer-reviewed", "clinical", "independent"]

    # --- REAL templates: factual, neutral, attribution-based ---
    real_templates = [
        "{company} reports {quarter} quarter earnings of ${amount} billion",
        "{committee_name} committee approves {topic} bill with bipartisan support",
        "Study finds {finding} in {field} research published in {journal}",
        "{city} mayor announces ${amount} million {topic} infrastructure initiative",
        "International summit addresses {topic} trade and security concerns",
        "Supreme Court to hear oral arguments in {topic} case next term",
        "{country} and {country2} sign bilateral trade agreement on {topic}",
        "Federal Reserve signals {percent} percent interest rate {adjustment}",
        "University of {city2} researchers publish {topic} findings",
        "Department of {dept} releases updated {topic} guidelines for 2024",
        "{company} announces expansion into {region} with ${amount} billion investment",
        "National weather service issues {weather_event} advisory for {region}",
        "Economic indicators suggest {trend} in consumer {metric}",
        "Healthcare officials recommend {action2} ahead of {season} season",
        "Local school board approves new {topic} curriculum for fall semester",
        "Transportation authority unveils {topic} improvement plan for {region}",
        "Environmental protection agency reports progress on {topic} goals",
        "{company} to invest ${amount} billion in {topic} over five years",
        "Census data reveals {trend} population shifts across {region}",
        "New {topic} regulations set to take effect in {year}",
        "Central bank maintains current {policy2} rate amid {economic2} outlook",
        "{country} parliament debates new {topic} legislation",
        "Major {company2} merger receives regulatory approval from {dept}",
        "Research team at {institution} develops breakthrough in {topic}",
        "City council votes to allocate ${amount} million for {topic}",
        "Export data shows {percent} percent increase in {topic} trade",
        "Industry report projects {trend} growth in {topic} sector",
        "{institution} releases annual {topic} report showing {trend} trends",
        "Lawmakers propose bipartisan {topic} reform to address {issue}",
        "{company} quarterly report shows {percent} percent revenue {adjustment2}",
        "Global {topic} conference attracts delegates from {number} countries",
        "Public health agency updates {topic} guidelines based on new data",
        "Housing market shows {trend} in {region} metropolitan areas",
        "{institution} study examines long-term effects of {topic}",
        "State governor signs {topic} bill into law at {event}",
        "Energy sector reports {percent} percent increase in {topic} output",
        "Technology firms collaborate on new {topic} initiative",
        "Central bank governor addresses {topic} concerns at press conference",
        "National {topic} index shows {trend} for third consecutive quarter",
        "Research {topic2} finds correlation between {factor1} and {factor2}",
    ]

    companies = ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Meta",
                 "Samsung", "Intel", "Boeing", "Toyota", "Johnson & Johnson",
                 "Pfizer", "JPMorgan", "Walmart", "ExxonMobil"]
    companies2 = ["technology", "pharmaceutical", "energy", "financial",
                  "automotive", "retail", "aerospace"]
    quarters = ["first", "second", "third", "fourth", "fiscal"]
    cities = ["New York", "Chicago", "Los Angeles", "Houston", "Phoenix",
              "Seattle", "Denver", "Atlanta", "Boston", "Portland", "Austin"]
    countries = ["United States", "Canada", "United Kingdom", "Germany", "Japan",
                 "Australia", "France", "South Korea", "India", "Brazil"]
    topics = ["immigration", "education", "healthcare", "cybersecurity", "climate",
              "housing", "transportation", "technology", "energy", "agriculture",
              "trade", "defense", "infrastructure", "finance", "telecommunications"]
    findings = ["significant progress", "notable improvements", "new correlations",
                "potential risks", "measurable outcomes", "emerging patterns",
                "statistical significance", "preliminary results"]
    fields = ["medical", "environmental", "economic", "social", "nutritional",
              "engineering", "psychological", "epidemiological"]
    journals = ["Nature", "The Lancet", "Science", "JAMA", "PNAS",
                "Cell", "The BMJ", "New England Journal of Medicine"]
    depts = ["Education", "Health", "Energy", "Commerce", "Transportation",
             "Labor", "Defense", "Agriculture", "Justice", "Interior"]
    regions = ["the Midwest", "the Northeast", "the Pacific Northwest",
               "the Southeast", "the Southwest", "Northern Europe", "East Asia",
               "Southeast Asia", "Latin America", "sub-Saharan Africa"]
    trends = ["an increase", "a decrease", "continued stability", "a slight uptick",
              "a notable shift", "steady growth", "a gradual decline", "moderate expansion"]
    metrics = ["spending", "investment", "production", "employment", "output",
               "confidence", "activity", "sentiment"]
    actions2 = ["vaccination", "preventive measures", "updated precautions",
                "booster doses", "enhanced screening", "early detection"]
    seasons = ["winter", "flu", "allergy", "respiratory"]
    events = ["a signing ceremony", "a press conference", "an official event"]
    policy2 = ["monetary", "interest rate", "lending", "base rate"]
    economic2 = ["mixed", "cautiously optimistic", "moderately positive",
                 "stable", "uncertain"]
    issues = ["rising costs", "access disparities", "system efficiency",
              "public safety", "environmental impact"]
    adjustments = ["cut", "hike", "adjustment", "revision", "reduction"]
    adjustments2 = ["growth", "decline", "increase", "expansion", "improvement"]
    numbers = ["40", "60", "80", "100", "120", "30"]
    years = ["2024", "2025", "next year", "the coming fiscal year"]
    institutions = ["MIT", "Harvard", "Stanford", "Oxford", "Cambridge",
                    "Johns Hopkins", "ETH Zurich", "the University of Tokyo"]
    factors1 = ["diet", "exercise", "sleep patterns", "air quality", "stress levels",
                "income levels", "education access"]
    factors2 = ["health outcomes", "economic mobility", "academic performance",
                "disease risk", "quality of life", "workplace productivity"]
    weather_events = ["severe weather", "winter storm", "heat wave", "flood",
                      "hurricane", "wildfire smoke"]
    issues_list = ["supply chain", "public access", "workforce", "safety",
                   "quality assurance", "cost reduction"]
    committee_names = ["Senate Judiciary", "Senate Finance", "Senate Armed Services",
                       "House Energy", "House Appropriations", "Senate Commerce",
                       "House Intelligence", "Senate Foreign Relations"]

    headlines = []

    # Generate ~1500 fake headlines with realistic patterns
    for _ in range(1500):
        template = random.choice(fake_templates)
        headline = template.format(
            role=random.choice(roles),
            group=random.choice(groups),
            org=random.choice(orgs),
            topic=random.choice(topics_health + topics),
            health_problem=random.choice(health_problems),
            food=random.choice(foods),
            treatment=random.choice(treatments),
            industry=random.choice(industries),
            politician=random.choice(politicians),
            policy=random.choice(policies),
            economic_problem=random.choice(economic_problems),
            economic_action=random.choice(economic_actions),
            vaccine=random.choice(vaccines),
            study=random.choice(studies),
            action="secretly approving backroom deals",
        )
        headlines.append((headline, 'FAKE'))

    # Generate ~1500 real headlines
    for _ in range(1500):
        template = random.choice(real_templates)
        headline = template.format(
            company=random.choice(companies),
            company2=random.choice(companies2),
            city=random.choice(cities),
            city2=random.choice(cities),
            country=random.choice(countries),
            country2=random.choice([c for c in countries if c != "United States"]),
            topic=random.choice(topics),
            topic2=random.choice(topics),
            finding=random.choice(findings),
            field=random.choice(fields),
            journal=random.choice(journals),
            percent=random.randint(1, 15),
            amount=random.randint(1, 50),
            dept=random.choice(depts),
            region=random.choice(regions),
            trend=random.choice(trends),
            metric=random.choice(metrics),
            quarter=random.choice(quarters),
            action2=random.choice(actions2),
            season=random.choice(seasons),
            policy2=random.choice(policy2),
            economic2=random.choice(economic2),
            event=random.choice(events),
            committee_name=random.choice(committee_names),
            adjustment=random.choice(adjustments),
            adjustment2=random.choice(adjustments2),
            number=random.choice(numbers),
            year=random.choice(years),
            institution=random.choice(institutions),
            factor1=random.choice(factors1),
            factor2=random.choice(factors2),
            weather_event=random.choice(weather_events),
            issue=random.choice(issues_list),
        )
        headlines.append((headline, 'REAL'))

    random.shuffle(headlines)

    with open(DATASET_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['headline', 'label'])
        writer.writerows(headlines)

    print(f"Synthetic dataset generated: {len(headlines)} headlines saved to {DATASET_PATH}")
    return DATASET_PATH


def load_real_dataset():
    """Load the real fake news dataset (ISOT / fake_or_real_news.csv).

    Uses the 'title' field as headline. Returns list of (headline, label) tuples.
    """
    headlines = []
    if not os.path.exists(REAL_DATASET_PATH):
        print(f"Real dataset not found at {REAL_DATASET_PATH}, skipping.")
        return headlines

    with open(REAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use title as headline (the app takes short text input)
            title = row.get('title', '').strip()
            label = row.get('label', '').strip().upper()
            if title and label in ('FAKE', 'REAL'):
                headlines.append((title, label))

    print(f"Loaded {len(headlines)} articles from real dataset")
    return headlines


def train_model():
    """Train the fake news detection model on real + synthetic data."""
    # Generate synthetic dataset
    generate_dataset()

    # Load synthetic data
    synthetic = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            synthetic.append((row['headline'], row['label']))

    # Load real data
    real = load_real_dataset()

    # Combine: real data takes priority, synthetic adds diversity
    all_headlines = real + synthetic
    print(f"Total training data: {len(all_headlines)} ({len(real)} real + {len(synthetic)} synthetic)")

    headlines = [h for h, _ in all_headlines]
    labels = [l for _, l in all_headlines]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        headlines, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Create TF-IDF vectorizer with better parameters
    vectorizer = TfidfVectorizer(
        max_features=15000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Logistic Regression with tuned hyperparameters
    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    accuracy = model.score(X_test_tfidf, y_test)
    print(f"Model trained — Accuracy: {accuracy:.2%}")

    # Per-class accuracy
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test_tfidf)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

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

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model: {e}. Retraining...")
        return train_model()


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
    model, vectorizer = train_model()

    # Test predictions
    test_headlines = [
        "BREAKING: Government hiding aliens from the public!!!",
        "Federal Reserve announces 3% interest rate adjustment",
        "You WON'T BELIEVE what the President did to citizens!!!",
        "University researchers publish findings on climate",
        "SHOCKING: Vaccines causes widespread panic — doctors are STUNNED",
        "Indian government announces new measures to promote electric vehicle adoption across major cities",
        "Exposed: The hidden agenda behind vaccine safety that no one is talking about",
        "Senate passes bipartisan cybersecurity reform bill",
        "Doctors do not want you to know about this treatment for heart disease",
        "Apple reports first quarter earnings of $12 billion",
    ]

    print("\n--- Test Predictions ---")
    for h in test_headlines:
        result = predict(h, model, vectorizer)
        print(f"  [{result['label']}] ({result['confidence']:.1f}%) {h[:70]}...")
