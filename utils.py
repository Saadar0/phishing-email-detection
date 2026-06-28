# utils.py
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Full original keyword sets
        self.urgency_keywords = ['urgent', 'immediately', 'asap', 'expires', 'expire', 'expiring', 'limited time',
                                 'act now', 'hurry', 'last chance', 'final notice', 'time sensitive',
                                 'response required', 'action required', 'within 24 hours']
        self.threat_keywords = ['suspend', 'suspended', 'restricted', 'restriction', 'locked', 'lock', 'close',
                                'closed', 'closing', 'terminate', 'terminated', 'deactivate', 'unusual activity',
                                'suspicious activity', 'unauthorized access', 'security alert', 'fraud alert']
        self.action_keywords = ['verify', 'confirm', 'update', 'validate', 'authenticate', 'authorize', 'click here',
                                'click now', 'click below', 'click link', 'follow link', 'login', 'log in', 'sign in',
                                'access', 'review', 'check']
        self.reward_keywords = ['winner', 'won', 'prize', 'reward', 'claim', 'congratulations', 'selected', 'chosen',
                                'free', 'gift', 'bonus', 'promotion', 'refund', 'compensation', 'payment']
        self.financial_keywords = ['bank', 'account', 'credit card', 'debit card', 'paypal', 'payment', 'transaction',
                                   'invoice', 'billing', 'balance', 'funds', 'money', 'transfer', 'wire', 'deposit']
        self.brand_keywords = ['paypal', 'amazon', 'ebay', 'microsoft', 'apple', 'google', 'facebook', 'netflix',
                               'bank of america', 'wells fargo', 'chase', 'irs', 'fedex', 'ups', 'dhl']
        self.personal_info_keywords = ['password', 'ssn', 'social security', 'credit card number', 'card number', 'cvv',
                                       'pin', 'account number', 'routing number', 'date of birth', 'mother maiden',
                                       'security question']
        self.fear_words = ['warning', 'alert', 'unauthorized', 'suspicious', 'fraud', 'theft', 'stolen', 'hacked',
                           'breach', 'compromise']
        self.greed_words = ['win', 'free', 'bonus', 'gift', 'money', 'cash', 'reward', 'prize', 'million', 'thousand']
        self.fraud_signals = ['million dollars', 'inheritance', 'next of kin', 'transfer funds', 'confidential',
                              'god bless', 'foreign transfer', 'dying', 'cancer', 'widow', 'prince', 'diplomat',
                              'consignment']
        self.time_phrases = ['within 24 hours', 'within 48 hours', 'by midnight', 'by today', 'deadline', 'expire']
        self.digital_ctas = self.action_keywords + ['click', 'here', 'now', 'link']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            return X.apply(self._clean_text)
        return [self._clean_text(t) for t in X]

    def _clean_text(self, text):
        if pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', ' [url] ', text)
        text = re.sub(r'\S+@\S+', ' [email] ', text)
        text = re.sub(r'[\+\d\-\s\(\)]{10,}', ' [phone] ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\$\%\[\]\-\']', ' ', text)
        text = re.sub(r'\b\d+\b', ' [number] ', text)
        return ' '.join(text.split())

    def _extract_urls(self, raw_text):
        pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
        return re.findall(pattern, str(raw_text))

    def _url_features(self, urls):
        feats = {'url_count': len(urls), 'has_url': int(len(urls) > 0), 'url_length_avg': 0, 'url_length_max': 0,
                 'url_has_ip': 0, 'url_suspicious_tld': 0, 'url_has_at_symbol': 0, 'url_subdomain_count_avg': 0,
                 'url_is_shortened': 0, 'url_hyphen_in_domain': 0}
        if not urls: return feats

        lengths = [len(u) for u in urls]
        feats['url_length_avg'] = float(np.mean(lengths))
        feats['url_length_max'] = int(max(lengths))

        # URL Logic
        ip_pat = re.compile(r'\d{1,3}(?:\.\d{1,3}){3}')
        suspicious_tlds = {'.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work', '.click', '.link', '.loan',
                           '.download'}
        for url in urls:
            if ip_pat.search(url): feats['url_has_ip'] = 1
            if any(url.lower().endswith(t) for t in suspicious_tlds): feats['url_suspicious_tld'] = 1
            if '@' in url: feats['url_has_at_symbol'] = 1
        return feats

    def extract_features(self, cleaned_text, original_text=None, return_ui_data=False):
        if original_text is None: original_text = cleaned_text
        feats = {}
        t = cleaned_text.lower()
        orig = str(original_text).lower()

        # ─── 1. Map ALL categories for UI ───
        ui_highlights = {
            "Urgency": [k for k in self.urgency_keywords if k in t],
            "Threats": [k for k in self.threat_keywords if k in t],
            "Actions": [k for k in self.action_keywords if k in t],
            "Rewards": [k for k in self.reward_keywords if k in t],
            "Financial": [k for k in self.financial_keywords if k in t],
            "Brands": [k for k in self.brand_keywords if k in t],
            "Personal Info": [k for k in self.personal_info_keywords if k in t],
            "Fraud Signals": [k for k in self.fraud_signals if k in t],
        }

        # ─── 2. Identify specific Mixed Alphanumeric words ───
        mix_patterns = re.findall(r'\b\w*\d+[a-zA-Z]+\w*\b|\b\w*[a-zA-Z]+\d+\w*\b', original_text)
        detected_mixed = [p for p in mix_patterns if p.lower() not in ['1st', '2nd', '3rd', '4th', 'th']]

        # ─── 3. Standard Feature Extraction (for the model) ───
        words = t.split()
        feats['text_length'] = len(t)
        feats['urgency_count'] = len(ui_highlights["Urgency"])
        feats['threat_count'] = len(ui_highlights["Threats"])
        feats['action_count'] = len(ui_highlights["Actions"])
        feats['reward_count'] = len(ui_highlights["Rewards"])
        feats['financial_count'] = len(ui_highlights["Financial"])
        feats['brand_count'] = len(ui_highlights["Brands"])
        feats['personal_info_count'] = len(ui_highlights["Personal Info"])
        feats['fraud_signal_count'] = len(ui_highlights["Fraud Signals"])
        feats['number_letter_mix'] = len(detected_mixed)

        # URL features
        urls = self._extract_urls(original_text)
        feats.update(self._url_features(urls))

        # Punctuation and Formatting
        feats['exclamation_count'] = orig.count('!')
        feats['question_count'] = orig.count('?')
        feats['excessive_exclamation'] = int('!!!' in orig)

        orig_words = str(original_text).split()
        caps_words = [w for w in orig_words if w.isupper() and len(w) > 2]
        feats['caps_word_ratio'] = len(caps_words) / len(orig_words) if orig_words else 0.0

        feats['repeated_chars'] = len(re.findall(r'(\w)\1{2,}', t))

        # Lexical diversity
        non_placeholder = [w for w in words if not (w.startswith('[') and w.endswith(']'))]
        feats['lexical_diversity'] = (
            len(set(non_placeholder)) / len(non_placeholder) if non_placeholder else 0.0
        )

        # Repetition ratio for CTAs
        cta_words = [w for w in words if w in self.digital_ctas]
        feats['cta_repetition_ratio'] = (
            (len(cta_words) - len(set(cta_words))) / len(cta_words) if cta_words else 0.0
        )

        if return_ui_data:
            return feats, ui_highlights, detected_mixed
        return feats