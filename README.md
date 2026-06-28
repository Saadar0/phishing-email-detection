# 🎯 PhishGuard AI

**PhishGuard AI** is a cybersecurity web application that leverages Machine Learning to detect phishing emails in real time. 

Unlike traditional filters that merely check if a sender is on a blacklist, this artificial intelligence performs deep semantic and behavioral analysis. It extracts the textual "DNA" of an email (detecting hidden links, linguistic urgency, digital calls-to-action, and financial anomalies like fake invoices) to calculate a precise risk score and highlight suspicious elements for the user.

---

## 🏗️ System Architecture

The architecture relies on a structured data processing pipeline broken down into 4 key stages:

```text
[Raw Email Text] 
       │
       ▼
[1. utils.py (TextPreprocessor)] ──► Cleans text & extracts numerical features
       │
       ▼
[2. col_preprocessor.pkl] ─────────► Transforms text into vectors (TF-IDF) & normalizes numbers
       │
       ▼
[3. phishing_voting_model.pkl] ────► Voting-based classification (Naive Bayes + Random Forest)
       │
       ▼
[4. app.py (Streamlit)] ───────────► Graphical rendering of verdict and forensic highlighting

```

---

## 📂 File Structure & Roles

The project is designed with a modular and professional architecture, split across several strategic files:

* **`utils.py` (The Core Logic):** Contains the custom `TextPreprocessor` class. This is the engine that prepares raw text for the AI by handling text cleaning (using RegEx to standardize emails, phone numbers, and URLs into tokens like `[url]`, `[email]`) and feature extraction (counting phishing keywords, text length, and action requests).
* **`train.py` (The Training Workshop):** An offline script executed by the developer to train the system. It processes a large dataset of "Ham" (safe) and "Phishing" emails, uses `utils.py` to vectorize the data, and trains an ensemble *Voting Classifier* (combining *Multinomial Naive Bayes* and *Random Forest*). It then exports the trained brains as serialized `.pkl` files.
* **`app.py` (The User Interface):** The interactive web dashboard built with Streamlit. It features a premium dark-themed UI tailored for cyber-defense tools. When a user pastes an email, it calls the `.pkl` models in the background to instantly output a threat probability gauge and use `annotated_text` to color-code the email (red for dangerous keywords, green for safe context).

---

## 🚀 Installation & Usage

### Prerequisites

* Python 3.8 or higher

### 1. Clone the Repository

```bash
git clone https://github.com/Saadar0/phishing-email-detection.git
cd phishing-email-detection

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Train the Model (Generate .pkl Files)

Before running the web app, you need to generate the trained model files (which are excluded from the GitHub repository via `.gitignore` due to file size):

```bash
python train.py

```

### 4. Launch the Web Application

```bash
streamlit run app.py

```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

