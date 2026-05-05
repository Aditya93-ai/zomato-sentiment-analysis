# ============================================================
# ZOMATO RESTAURANT REVIEWS — SENTIMENT ANALYSIS
# Author: Aditya | BCA Graduate | Data Analytics Portfolio
# Tools: Python, VADER, TF-IDF, Scikit-learn, Matplotlib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import re
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
BG     = '#0D1117'
CARD   = '#161B22'
POS    = '#2ECC71'
NEG    = '#E74C3C'
NEU    = '#F39C12'
ACCENT = '#E23744'   # Zomato red
TEXT   = '#E6EDF3'
MUTED  = '#8B949E'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'axes.edgecolor': '#30363D', 'axes.labelcolor': TEXT,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': TEXT, 'grid.color': '#21262D',
    'grid.alpha': 0.4, 'font.family': 'monospace',
})

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('/home/claude/zomato_sentiment/zomato_reviews.csv')
print("=" * 58)
print("  ZOMATO SENTIMENT ANALYSIS")
print("=" * 58)
print(f"\n📦 Dataset: {df.shape[0]} reviews across {df['city'].nunique()} cities")
print(f"🍽️  Restaurants: {df['restaurant_name'].nunique()}")
print(f"📊 Avg Rating: {df['rating'].mean():.2f} / 5.0")
print(f"\nSentiment Distribution:")
print(df['sentiment_label'].value_counts())

# ─────────────────────────────────────────────
# 2. TEXT CLEANING
# ─────────────────────────────────────────────
STOPWORDS = {
    'i','me','my','the','a','an','and','or','but','in','on','at','to',
    'for','of','with','is','was','it','this','that','be','are','were',
    'have','had','has','do','did','not','so','just','very','will',
    'would','could','should','they','them','their','we','our','you',
    'your','he','she','his','her','its','as','by','from','up','about',
    'into','than','then','when','what','which','who','how','all','been',
    'also','there','here','more','out','get','got','after','before',
    'over','no','any','some','one','two','can','may','even','too'
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

df['clean_review'] = df['review_text'].apply(clean_text)
print(f"\n✅ Text cleaned — sample:")
print(f"   Original : {df['review_text'].iloc[0][:60]}...")
print(f"   Cleaned  : {df['clean_review'].iloc[0][:60]}...")

# ─────────────────────────────────────────────
# 3. VADER SENTIMENT SCORING
# ─────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive', compound
    elif compound <= -0.05:
        return 'Negative', compound
    else:
        return 'Neutral', compound

vader_results = df['review_text'].apply(get_vader_sentiment)
df['vader_sentiment'] = vader_results.apply(lambda x: x[0])
df['vader_compound']  = vader_results.apply(lambda x: x[1])

# VADER accuracy vs our labels
vader_acc = (df['vader_sentiment'] == df['sentiment_label']).mean()
print(f"\n📈 VADER Accuracy vs ground truth labels: {vader_acc*100:.1f}%")

# ─────────────────────────────────────────────
# 4. EDA DASHBOARD
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor(BG)
fig.suptitle('ZOMATO REVIEWS — EXPLORATORY DATA ANALYSIS',
             fontsize=17, fontweight='bold', color=TEXT, y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Sentiment Distribution
ax1 = fig.add_subplot(gs[0, 0])
counts = df['sentiment_label'].value_counts()
colors = [POS, NEG, NEU]
labels = counts.index.tolist()
wedges, texts, autotexts = ax1.pie(
    counts, labels=labels, colors=colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.62, edgecolor=BG, linewidth=3),
    textprops={'color': TEXT, 'fontsize': 11})
for at in autotexts:
    at.set_fontweight('bold'); at.set_fontsize(12)
ax1.set_title('Overall Sentiment Split', color=ACCENT, fontweight='bold', pad=15)

# Plot 2: Rating vs Sentiment
ax2 = fig.add_subplot(gs[0, 1])
rating_sent = df.groupby(['rating', 'sentiment_label']).size().unstack(fill_value=0)
x = np.arange(len(rating_sent))
w = 0.26
ax2.bar(x - w, rating_sent.get('Positive', 0), w, color=POS, label='Positive', edgecolor=BG)
ax2.bar(x,     rating_sent.get('Neutral', 0),  w, color=NEU, label='Neutral',  edgecolor=BG)
ax2.bar(x + w, rating_sent.get('Negative', 0), w, color=NEG, label='Negative', edgecolor=BG)
ax2.set_xticks(x); ax2.set_xticklabels([f'★{r}' for r in rating_sent.index])
ax2.set_title('Rating vs Sentiment', color=ACCENT, fontweight='bold', pad=15)
ax2.set_ylabel('Review Count', color=MUTED)
ax2.legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Sentiment by City
ax3 = fig.add_subplot(gs[0, 2])
city_sent = df.groupby('city')['sentiment_label'].apply(
    lambda x: (x == 'Positive').mean() * 100).sort_values(ascending=True)
colors_city = [POS if v >= 60 else NEU if v >= 45 else NEG for v in city_sent.values]
bars = ax3.barh(city_sent.index, city_sent.values, color=colors_city,
                edgecolor=BG, linewidth=1.5, height=0.6)
for bar, val in zip(bars, city_sent.values):
    ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9, color=TEXT, fontweight='bold')
ax3.set_title('Positive Sentiment % by City', color=ACCENT, fontweight='bold', pad=15)
ax3.set_xlabel('Positive Review Rate (%)', color=MUTED)
ax3.grid(axis='x', alpha=0.3)
ax3.set_xlim(0, 100)

# Plot 4: Delivery Time vs Sentiment
ax4 = fig.add_subplot(gs[1, 0])
for sent, col in [('Positive', POS), ('Neutral', NEU), ('Negative', NEG)]:
    data = df[df['sentiment_label'] == sent]['delivery_time_mins']
    ax4.hist(data, bins=20, alpha=0.65, color=col, label=sent, edgecolor=BG)
ax4.set_title('Delivery Time vs Sentiment', color=ACCENT, fontweight='bold', pad=15)
ax4.set_xlabel('Delivery Time (mins)', color=MUTED)
ax4.set_ylabel('Count', color=MUTED)
ax4.legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
ax4.grid(alpha=0.3)
pos_mean = df[df['sentiment_label']=='Positive']['delivery_time_mins'].mean()
neg_mean = df[df['sentiment_label']=='Negative']['delivery_time_mins'].mean()
ax4.axvline(pos_mean, color=POS, linestyle='--', linewidth=2, alpha=0.8)
ax4.axvline(neg_mean, color=NEG, linestyle='--', linewidth=2, alpha=0.8)

# Plot 5: Order Value vs Sentiment
ax5 = fig.add_subplot(gs[1, 1])
sent_order = df.groupby('sentiment_label')['order_value_inr'].mean().reindex(['Positive','Neutral','Negative'])
colors5 = [POS, NEU, NEG]
bars5 = ax5.bar(sent_order.index, sent_order.values, color=colors5,
                edgecolor=BG, linewidth=2, width=0.5)
for bar, val in zip(bars5, sent_order.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'₹{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11, color=TEXT)
ax5.set_title('Avg Order Value vs Sentiment', color=ACCENT, fontweight='bold', pad=15)
ax5.set_ylabel('Avg Order Value (₹)', color=MUTED)
ax5.grid(axis='y', alpha=0.3)

# Plot 6: VADER compound score distribution
ax6 = fig.add_subplot(gs[1, 2])
for sent, col in [('Positive', POS), ('Neutral', NEU), ('Negative', NEG)]:
    data = df[df['sentiment_label'] == sent]['vader_compound']
    ax6.hist(data, bins=25, alpha=0.65, color=col, label=f'{sent}', edgecolor=BG)
ax6.axvline(0.05,  color='white', linestyle='--', linewidth=1.5, alpha=0.5)
ax6.axvline(-0.05, color='white', linestyle='--', linewidth=1.5, alpha=0.5)
ax6.set_title('VADER Compound Score Distribution', color=ACCENT, fontweight='bold', pad=15)
ax6.set_xlabel('Compound Score (-1 to +1)', color=MUTED)
ax6.set_ylabel('Count', color=MUTED)
ax6.legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
ax6.grid(alpha=0.3)

plt.savefig('/home/claude/zomato_sentiment/01_eda_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("\n✅ EDA Dashboard saved")

# ─────────────────────────────────────────────
# 5. ML MODEL — TF-IDF + CLASSIFICATION
# ─────────────────────────────────────────────
print("\n📊 Training ML Models...")

le = LabelEncoder()
y = le.fit_transform(df['sentiment_label'])  # Negative=0, Neutral=1, Positive=2

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
X = tfidf.fit_transform(df['clean_review'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Naive Bayes
nb = MultinomialNB(alpha=0.5)
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_nb = accuracy_score(y_test, y_pred_nb)
print(f"\n📈 Logistic Regression Accuracy : {acc_lr*100:.2f}%")
print(f"📈 Naive Bayes Accuracy         : {acc_nb*100:.2f}%")
print(f"\n📋 Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# ─────────────────────────────────────────────
# 6. MODEL PERFORMANCE DASHBOARD
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.patch.set_facecolor(BG)
fig.suptitle('SENTIMENT CLASSIFICATION — MODEL PERFORMANCE',
             fontsize=16, fontweight='bold', color=TEXT, y=1.01)

# Confusion Matrix — LR
ax = axes[0]
cm = confusion_matrix(y_test, y_pred_lr)
class_names = le.classes_
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=2, linecolor=BG,
            annot_kws={'size': 13, 'weight': 'bold'})
ax.set_title('Confusion Matrix\n(Logistic Regression)', color=ACCENT, fontweight='bold', pad=15)
ax.set_ylabel('Actual', color=MUTED)
ax.set_xlabel('Predicted', color=MUTED)

# Model Comparison
ax = axes[1]
models = ['Logistic\nRegression', 'Naive\nBayes', 'VADER\n(Unsupervised)']
accs   = [acc_lr * 100, acc_nb * 100, vader_acc * 100]
colors_m = [ACCENT, '#3498DB', '#9B59B6']
bars = ax.bar(models, accs, color=colors_m, edgecolor=BG, linewidth=2, width=0.5)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold',
            fontsize=12, color=TEXT)
ax.set_title('Model Accuracy Comparison', color=ACCENT, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy (%)', color=MUTED)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)
ax.axhline(50, color=MUTED, linestyle='--', alpha=0.4, linewidth=1.5)
ax.text(2.4, 51.5, 'random baseline', color=MUTED, fontsize=8)

# Top TF-IDF features per class
ax = axes[2]
feature_names = np.array(tfidf.get_feature_names_out())
top_n = 8
y_pos = 0
tick_pos, tick_labels = [], []
colors_feat = {'Positive': POS, 'Negative': NEG, 'Neutral': NEU}

for idx, class_name in enumerate(le.classes_):
    coef = lr.coef_[idx]
    top_idx = np.argsort(coef)[-top_n:]
    top_words = feature_names[top_idx]
    top_vals  = coef[top_idx]
    col = colors_feat[class_name]
    for word, val in zip(top_words, top_vals):
        ax.barh(y_pos, val, color=col, edgecolor=BG, height=0.7, alpha=0.85)
        ax.text(-0.05, y_pos, word, ha='right', va='center', fontsize=7.5,
                color=TEXT)
        tick_pos.append(y_pos)
        y_pos += 1
    y_pos += 1.5

ax.set_title('Top Keywords by Sentiment Class\n(TF-IDF Logistic Regression)',
             color=ACCENT, fontweight='bold', pad=15)
ax.set_xlabel('Feature Coefficient', color=MUTED)
ax.set_yticks([])
ax.grid(axis='x', alpha=0.3)
patches = [mpatches.Patch(color=c, label=l) for l, c in colors_feat.items()]
ax.legend(handles=patches, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/zomato_sentiment/02_model_performance.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ Model Performance Dashboard saved")

# ─────────────────────────────────────────────
# 7. ASPECT ANALYSIS — What are people talking about?
# ─────────────────────────────────────────────
aspects = {
    'Delivery Speed':  ['delivery', 'fast', 'quick', 'late', 'slow', 'time', 'minutes', 'arrived', 'delay'],
    'Food Quality':    ['taste', 'delicious', 'fresh', 'stale', 'quality', 'flavour', 'cooked', 'raw', 'amazing', 'terrible'],
    'Packaging':       ['packaging', 'packed', 'spilled', 'sealed', 'broken', 'neat', 'secure'],
    'Quantity':        ['quantity', 'portion', 'less', 'generous', 'small', 'enough'],
    'Price/Value':     ['price', 'value', 'expensive', 'cheap', 'worth', 'reasonable', 'overpriced'],
    'Customer Service':['support', 'refund', 'response', 'rude', 'polite', 'helpful', 'care'],
}

aspect_sentiment = {}
for aspect, keywords in aspects.items():
    pattern = '|'.join(keywords)
    mask = df['review_text'].str.lower().str.contains(pattern, na=False)
    subset = df[mask]
    if len(subset) > 0:
        pos_pct = (subset['sentiment_label'] == 'Positive').mean() * 100
        neg_pct = (subset['sentiment_label'] == 'Negative').mean() * 100
        aspect_sentiment[aspect] = {
            'count': len(subset), 'positive': pos_pct, 'negative': neg_pct
        }

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.patch.set_facecolor(BG)
fig.suptitle('ASPECT-BASED SENTIMENT ANALYSIS — WHAT CUSTOMERS TALK ABOUT',
             fontsize=15, fontweight='bold', color=TEXT, y=1.01)

# Aspect mention frequency
ax = axes[0]
asp_names  = list(aspect_sentiment.keys())
asp_counts = [aspect_sentiment[a]['count'] for a in asp_names]
sorted_idx = np.argsort(asp_counts)
asp_names_s  = [asp_names[i] for i in sorted_idx]
asp_counts_s = [asp_counts[i] for i in sorted_idx]
colors_asp = [ACCENT if c == max(asp_counts_s) else '#3498DB' for c in asp_counts_s]
bars = ax.barh(asp_names_s, asp_counts_s, color=colors_asp,
               edgecolor=BG, linewidth=1.5, height=0.6)
for bar, val in zip(bars, asp_counts_s):
    ax.text(val + 5, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=10, color=TEXT, fontweight='bold')
ax.set_title('Most Discussed Aspects\n(Review Mention Count)', color=ACCENT, fontweight='bold', pad=15)
ax.set_xlabel('Number of Reviews Mentioning Aspect', color=MUTED)
ax.grid(axis='x', alpha=0.3)

# Positive vs Negative % per aspect
ax = axes[1]
x = np.arange(len(asp_names))
w = 0.35
pos_vals = [aspect_sentiment[a]['positive'] for a in asp_names]
neg_vals = [aspect_sentiment[a]['negative'] for a in asp_names]
ax.bar(x - w/2, pos_vals, w, color=POS, label='Positive %', edgecolor=BG, linewidth=1.5)
ax.bar(x + w/2, neg_vals, w, color=NEG, label='Negative %', edgecolor=BG, linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(asp_names, rotation=20, ha='right', fontsize=9)
ax.set_title('Sentiment Split per Aspect', color=ACCENT, fontweight='bold', pad=15)
ax.set_ylabel('Percentage of Reviews (%)', color=MUTED)
ax.legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/zomato_sentiment/03_aspect_analysis.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ Aspect Analysis Dashboard saved")

# ─────────────────────────────────────────────
# 8. BUSINESS SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("  BUSINESS INSIGHTS SUMMARY")
print("=" * 58)
best_city  = city_sent.idxmax()
worst_city = city_sent.idxmin()
pos_del = df[df['sentiment_label']=='Positive']['delivery_time_mins'].mean()
neg_del = df[df['sentiment_label']=='Negative']['delivery_time_mins'].mean()
most_aspect = max(aspect_sentiment, key=lambda a: aspect_sentiment[a]['count'])

print(f"\n🏙️  Best city (positive sentiment)  : {best_city} ({city_sent.max():.1f}%)")
print(f"🏙️  Worst city (positive sentiment) : {worst_city} ({city_sent.min():.1f}%)")
print(f"\n⏱️  Avg delivery — Positive reviews : {pos_del:.1f} mins")
print(f"⏱️  Avg delivery — Negative reviews : {neg_del:.1f} mins")
print(f"\n💬 Most discussed aspect            : {most_aspect}")
print(f"\n📈 Logistic Regression Accuracy     : {acc_lr*100:.1f}%")
print(f"📈 Naive Bayes Accuracy             : {acc_nb*100:.1f}%")
print(f"📈 VADER Accuracy                   : {vader_acc*100:.1f}%")
print("=" * 58)
