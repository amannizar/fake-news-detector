"""Fake News Detector — Flask Application."""

import os
import secrets
from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, AnalysisHistory
from ml_model import load_model, predict

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'fakenews.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access the news analyzer.'
login_manager.login_message_category = 'info'

# Load ML model globally
ml_model = None
ml_vectorizer = None


def get_model():
    """Lazy-load the ML model."""
    global ml_model, ml_vectorizer
    if ml_model is None or ml_vectorizer is None:
        ml_model, ml_vectorizer = load_model()
    return ml_model, ml_vectorizer


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Landing page — redirect to analyzer or login."""
    if current_user.is_authenticated:
        return redirect(url_for('analyze'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration."""
    if current_user.is_authenticated:
        return redirect(url_for('analyze'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        # Validation
        errors = []
        if not username or len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if not email or '@' not in email:
            errors.append('Please enter a valid email address.')
        if not password or len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm:
            errors.append('Passwords do not match.')
        if User.query.filter_by(username=username).first():
            errors.append('Username already taken.')
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered.')

        if errors:
            for e in errors:
                flash(e, 'error')
            return render_template('register.html', username=username, email=email)

        # Create user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if current_user.is_authenticated:
        return redirect(url_for('analyze'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user, remember=True)
            flash('Welcome back!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('analyze'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Logout the current user."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    """Main news analyzer page."""
    if request.method == 'POST':
        headline = request.form.get('headline', '').strip()

        if not headline:
            flash('Please enter a news headline or article text.', 'error')
            return render_template('analyze.html')

        if len(headline) < 5:
            flash('Please enter at least 5 characters for analysis.', 'error')
            return render_template('analyze.html')

        # Run prediction
        model, vectorizer = get_model()
        result = predict(headline, model, vectorizer)

        # Save to history
        analysis = AnalysisHistory(
            user_id=current_user.id,
            headline=headline,
            result=result['label'],
            confidence=result['confidence']
        )
        db.session.add(analysis)
        db.session.commit()

        return redirect(url_for('result', analysis_id=analysis.id))

    return render_template('analyze.html')


@app.route('/result/<int:analysis_id>')
@login_required
def result(analysis_id):
    """Display analysis result."""
    analysis = AnalysisHistory.query.filter_by(
        id=analysis_id, user_id=current_user.id
    ).first_or_404()
    return render_template('result.html', analysis=analysis)


@app.route('/history')
@login_required
def history():
    """Show analysis history for current user."""
    page = request.args.get('page', 1, type=int)
    analyses = AnalysisHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(
        AnalysisHistory.analyzed_at.desc()
    ).paginate(page=page, per_page=10, error_out=False)
    return render_template('history.html', analyses=analyses)


@app.route('/history/delete/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    """Delete a history item."""
    analysis = AnalysisHistory.query.filter_by(
        id=analysis_id, user_id=current_user.id
    ).first_or_404()
    db.session.delete(analysis)
    db.session.commit()
    flash('Analysis deleted.', 'info')
    return redirect(url_for('history'))


# ---------------------------------------------------------------------------
# Initialize and run
# ---------------------------------------------------------------------------

with app.app_context():
    db.create_all()
    # Pre-load the model at startup
    get_model()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
