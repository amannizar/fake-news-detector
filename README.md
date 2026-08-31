# 🛡️ FakeGuard — AI-Powered Fake News Detector

A full-stack web application that detects fake news using Machine Learning. Paste any news headline and get an instant **FAKE** or **REAL** prediction with a confidence percentage.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-green?logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6-orange?logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Screenshots

### Login Page
![Login Page](screenshots/login.png)

### News Analyzer
![Analyzer Page](screenshots/analyze.png)

### Analysis Result
![Result Page](screenshots/result.png)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔐 **User Authentication** | Register & login with secure password hashing |
| 🔍 **News Analyzer** | Paste any headline or article text for instant analysis |
| 📊 **Confidence Score** | Animated gauge showing prediction confidence (0–100%) |
| 📜 **Analysis History** | All past analyses saved with pagination support |
| 🗑️ **History Management** | View details or delete past analyses |
| 🧠 **ML Model** | TF-IDF + Logistic Regression trained on 1000+ headlines |
| 🎨 **Premium Dark UI** | Glassmorphism, floating particles, neon gradients |
| 📱 **Responsive** | Works on desktop and mobile devices |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python Flask |
| **Database** | SQLite (via Flask-SQLAlchemy) |
| **ML Model** | Scikit-learn (TF-IDF Vectorizer + Logistic Regression) |
| **Authentication** | Flask-Login + Werkzeug password hashing |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Fonts** | Google Fonts (Inter, JetBrains Mono) |

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py                    # Flask application & routes
├── models.py                 # SQLAlchemy database models
├── ml_model.py               # ML training & prediction pipeline
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── templates/
│   ├── base.html             # Base layout with navigation
│   ├── login.html            # Login page
│   ├── register.html         # Registration page
│   ├── analyze.html          # News analyzer (main page)
│   ├── result.html           # Analysis result with gauge
│   └── history.html          # Analysis history list
│
├── static/
│   ├── css/style.css         # Premium dark theme styles
│   └── js/app.js             # Animations & interactions
│
├── screenshots/              # App screenshots for README
│
├── dataset/                  # Auto-generated on first run
│   └── news.csv              # 1000 synthetic labeled headlines
│
└── trained_model/            # Auto-generated on first run
    ├── model.pkl             # Trained classifier
    └── vectorizer.pkl        # TF-IDF vectorizer
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

> **Note:** On first run, the app will automatically:
> - Generate a synthetic dataset of 1000 headlines
> - Train the ML model
> - Create the SQLite database

---

## 🧠 How It Works

1. **Dataset Generation** — Creates 1000 synthetic headlines (500 fake + 500 real) using template patterns
2. **Feature Extraction** — Converts text to numerical features using TF-IDF Vectorization with bigrams
3. **Model Training** — Trains a Logistic Regression classifier on the features
4. **Prediction** — For new text input, transforms it using the trained vectorizer and predicts FAKE/REAL with confidence

### Fake News Patterns the Model Detects:
- ALL CAPS text and excessive punctuation (`!!!`)
- Sensationalist keywords (`BREAKING`, `SHOCKING`, `EXPOSED`)
- Conspiracy-style language (`they don't want you to know`)
- Clickbait patterns (`You WON'T BELIEVE`)

### Real News Patterns:
- Standard journalistic format
- Specific data points (percentages, dollar amounts)
- Institutional references (Federal Reserve, Supreme Court)
- Neutral, factual language

---

## 🔒 Security

- Passwords are hashed using Werkzeug's `generate_password_hash` (PBKDF2)
- Session management via Flask-Login
- CSRF protection through Flask's built-in session handling
- SQL injection prevention through SQLAlchemy ORM

---

## ⚠️ Disclaimer

This tool uses a simple ML model trained on synthetic data for **educational purposes only**. It should not be used as the sole basis for determining the veracity of any news article. Always verify information through multiple trusted sources.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Author

**Aman** — BCA Student & Developer

---

<p align="center">Made with ❤️ using Flask & Scikit-learn</p>
