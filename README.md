r/AITA Post Classifier & Web App Live Application Demo: 
[https://aita-classifier-marvin-xxxx.a.run.app](https://aita-classifier-482267030164.us-central1.run.app/)

An end-to-end Machine Learning microservice that classifies Reddit "Am I The Asshole?" (r/AITA) posts into YTA (Asshole) or NTA (Not The Asshole) verdicts using a fine-tuned Transformer model, served via a containerized FastAPI application on GCP Cloud Run.

## Executive Summary:

Redditors are commonly described as being a hive-mind, a singularly agreeing, predictable entity. This project started as an attempt to verify this claim. However, this quickly grew into an analysis of just how challenging the task of classifying interpersonal conflict bias in unstructured in social media text can be. From the perspective of the writing, social media posts are often noisy, grammatically and narratively inconsistent, and full of nuances like sarcasm. An additional layer of difficulty arises from the complex weaving of social norms, interpersonal relations, and financial obligations that r/AITA posts often describe. 

This project implements a production-grade NLP pipeline: 

1. **Fine-Tuned Transformer:** Fine-tuned RoBERTa on preprocessed r/AITA post submissions.
2. **Decision Threshold Calibration:** Evaluated decision boundaries across precision-recall trade-offs, confirming that the standard calibrated threshold $\tau = 0.50$ maximizes the $F_1$-score on validation data.
3. **Containerized Microservice:** Built using FastAPI and Docker, hardened for production, cached model weights for instant boot, and deployed to GCP Cloud Run.

## Decision Rule Formulation: 
Model predictions are governed by the calibrated probability threshold $\tau$:

$$\hat{y} = \begin{cases} 1 \text{ (YTA)}, & \text{if } P(Y = 1 \mid \mathbf{x}) \ge \tau \\ 0 \text{ (NTA)}, & \text{if } P(Y = 1 \mid \mathbf{x}) < \tau \end{cases}$$

where $\tau = 0.50$. Empirical threshold tuning confirmed that the uncalibrated probability output from softmax already yields the optimal harmonic balance between precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

## Error Analysis & Qualitative Insights

> 📖 **Deep Dive Notebook:** For full confusion matrices, scatter plots, and topic-level error distributions, see [`02_error_analysis.ipynb`](./notebooks/02_error_analysis.ipynb).

To evaluate model performance beyond aggregate metrics, we analyzed prediction failures across post length, topic clusters, and narrative structure. A clear pattern emerged: **text truncation drives systematic prediction bias**, while the model struggles with unwritten social contracts and emotional valence.

---

### 1. Text Length & Truncation Bias

Because long posts were truncated to fit the context window by retaining only the head and tail of the text, length played a major role in model errors:

```text
Post Length <= 150 words   ──►   "Trigger-Happy" (High FPR: 25.44%) ──► Over-predicts YTA
Post Length > 450 words    ──►   "Hesitant"      (High FNR: 49.77%) ──► Over-predicts NTA
```

- **Short Posts ($\le 150$ words):** Deprived of crucial nuance, the model becomes overly punitive. Lacking context, it reacts to isolated aggressive keywords, yielding a high False Positive Rate ($\text{FPR} = 25.44\%$), falsely accusing posters of being the asshole.
- **Long Posts ($> 450$ words):** Truncation severely degrades recall. The model defaults to the majority class (NTA), resulting in a massive False Negative Rate ($\text{FNR} = 49.77\%$), missing nearly half of all actual assholes. Furthermore, user edits, self-corrections, and "TL;DR" summaries at the end of long posts distort the model's temporal understanding of the story.

### 2. Topic Level Error Distributions
Grouping errors by topic clusters (constructed using Bertopic) reveals two distinct failure modes based on error bias ($\text{FNR} - \text{FPR}$):

#### Hesitant Topics (High False Negatives / Type 2 Errors)
**Topics: rent_lease_pay, work_shift_job, parking_park_car, roommates_roommate_room, dad_father_him, brother_he_him, christmas_family_birthday**

Behavior: The model frequently misses true assholes when posters justify their behavior using explicit rules, corporate policies, or legal rights (e.g., calling the police over a parking violation). While technically "within their rights," these actions often break unspoken social contracts. Additionally, drawn-out familial or roommate disputes make it difficult for the model to weigh retaliation versus self-defense.

#### "Trigger-Happy" Topics (High False Positives / Type 1 Errors)
**Topics: weight_gym_fat, wear_wearing_dress, lane_car_speed.**
Behavior: The model over-predicts YTA on short, emotionally charged posts containing high-valence trigger words (e.g., "fat", "screamed", "yelled", "inappropriate"). The model anchors heavily on the emotional tone rather than evaluating whether the reaction was justified in context.

### 3. Core Model Blindspots
1. **Intention vs. Impact:** The model struggles to separate a poster's benign intent from a harmful outcome. For instance, accidental harm caused while trying to join in on a joke is often misclassified because the model cannot weigh good intentions against negative results.
2. **Legal Rights vs. Unwritten Social Nuance:** The model favors written rules (e.g., employment contracts, parking enforcement) over implicit cultural consensus (e.g., workplace boundaries, destination wedding etiquette, driving courtesy).
3. **Gender & Topic Dynamics:** Topics centered around body image, clothing, or specific family dynamics (dad, brother) exhibit disproportionate error rates, pointing to potential domain-specific biases within the pre-trained weights or fine-tuning dataset.

🏗️ System ArchitecturePlaintext[ Reddit Post Input ] 
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ FastAPI App Container (GCP Cloud Run)                    │
 │  ├── Preprocessing (clean_and_format)                  │
 │  ├── Inference (RoBERTa PyTorch Model @ cache)          │
 │  └── Threshold Logic (YTA if P(Asshole) >= 0.50)       │
 └─────────────────────────────────────────────────────────┘
         │
         ▼
[ Reddit-Themed UI / JSON Prediction Response ]
🛠️ Tech StackMachine Learning & Modeling: PyTorch, Hugging Face Transformers, Scikit-learn, PandasAPI & Backend: FastAPI, Uvicorn, PydanticDevOps & Cloud: Docker, Google Cloud Platform (Cloud Run, Cloud Build, Artifact Registry)📁 Repository StructurePlaintext

aita_classifier/
├── .dockerignore            # Docker context exclusions
├── Dockerfile               # Production multi-stage Docker build
├── requirements.txt         # Lightweight production dependencies
├── requirements-dev.txt     # Local development & evaluation dependencies
├── README.md                # Project documentation
├── helpers.py
├── lookup.py
├── assets/                  # Images and visualizations for documentation
│   └── confusion_matrix.png
├── app/
│   ├── __init__.py          # Package marker
│   └── main.py              # FastAPI server, health endpoint & Reddit UI
├── notebooks/
│   └── 02_error_analysis.ipynb  # Error analysis & validation visualizations
└── src/
    └── train_modernbert.py  # Model training pipeline code

## Local Development Setup
1. Clone & Install Dependencies:

```Bash
git clone https://github.com/YOUR_USERNAME/r-AITA_classifier.git
cd r-AITA_classifier

# Create local environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development packages
pip install -r requirements-dev.txt
```

2. Run Application Locally:

```Bash
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000 in your browser to interact with the web interface.

## Docker Deployment
1. Build and Test Container Locally:
```Bash
docker build -t aita-classifier .
docker run -p 8000:8000 aita-classifier
```

2. Deploy to Google Cloud Run
```Bash
gcloud run deploy aita-classifier \
  --source . \
  --region us-central1 \
  --memory 2Gi \
  --port 8000 \
  --allow-unauthenticated
```
