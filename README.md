---
title: Flavorgen Cafe
emoji: ☕
colorFrom: yellow
colorTo: yellow
sdk: streamlit
app_file: app.py
pinned: false
---
# FlavorGen Café Lab — Hybrid AI Beverage Intelligence System

**Lakshmi Sruthi Anchula** · Applied Machine Learning Engineer · Full-Stack AI Systems Builder  
[LinkedIn](https://your-linkedin-link) · [GitHub](https://your-github-link) · [Live Demo on Hugging Face](https://your-hf-link)

---

## What This Project Does

FlavorGen Café Lab is a full-stack AI system that does two things most recommendation tools cannot: it finds the right drink for you based on how you actually describe your taste, and it invents a completely new recipe by blending two drinks you already like.

It is not a search bar. It is not a filter. It is a pipeline — from natural language input, through semantic similarity and flavor-vector mathematics, to a neural network that predicts ingredients the way a trained barista might think about combining flavors.

The system runs live on Hugging Face Spaces with MongoDB Atlas handling all user data persistence in the background.

---

## System Architecture

```
User Input (text query + taste preferences)
        │
        ▼
Hard Filter Layer
  type · temperature · sugar level · caffeine level
        │
        ▼
Hybrid Scoring Engine
  TF-IDF Vectorizer                Flavor Vector (8 dimensions)
  cosine_similarity(               cosine_similarity(
    query_vec, doc_mat)              user_flavor_vec, drink_flavor_vec)
        │                                      │
        └──────────── Weighted Fusion ─────────┘
          Score = 0.35 × TextSim + 0.65 × FlavorSim
        │
        ▼
Ranked Results + Explainability Breakdown + Radar Chart
        │
        ▼ (if Fusion mode selected)
FusionNet MLP (PyTorch)
  Input: fused flavor vectors of two parent drinks
  Output: Sigmoid probabilities across 50-ingredient vocabulary
  Final: Threshold → cohesive ingredient list + confidence scores
        │
        ▼
MongoDB Atlas
  favorites collection · session event log · user taste profile
```

---

## Dataset

The backbone of the system is `drinks_hybrid_augmented.csv` — a curated dataset of 50+ beverages, each encoded across 10 structured fields that serve both the hard filter layer and the vector computation engine.

![Dataset Structure](dataset_structure.jpg)

| Field | Type | Purpose |
|---|---|---|
| drink_id | Integer | Primary key, links to MongoDB event documents |
| name | String | Part of the TF-IDF corpus |
| type | Categorical | Hard filter — coffee, tea, smoothie, refresher |
| temperature | Categorical | Hard filter |
| sugar_level | Categorical | Hard filter |
| caffeine_level | Categorical | Hard filter |
| ingredient_ids | List of integers | Mapped to ingredient vocabulary for FusionNet |
| tags | List of strings | Augments TF-IDF document with semantic descriptors |
| popularity_score | Float | Secondary ranking signal |
| description | String | Primary TF-IDF document field |

The ingredient vocabulary covers 50 unique items across 8 flavor dimensions: sweet, bitter, creamy, fresh, fruity, nutty, warm spice, and acidic.

---

## Module 1 — Hybrid Recommendation Engine

**File:** `semantic_model.py` · **Library:** Scikit-Learn

### The Three-Stage Pipeline

**Stage 1 — Hard Filtering**

Before any machine learning runs, the candidate pool is narrowed by exact-match filtering on the user's stated preferences: drink type, temperature, sugar level, and caffeine level. This keeps similarity computation focused and fast.

**Stage 2 — TF-IDF Text Similarity**

Each drink's description and tags are concatenated into a single document and fitted into a `TfidfVectorizer`. When a user types a query, it is normalized and transformed into the same vector space. Cosine similarity is then computed between the query vector and every candidate document vector.

```python
q_vec = model.vectorizer.transform([q])
text_sim = cosine_similarity(q_vec, doc_mat).flatten()
text_sim = np.clip(text_sim, 0.0, 1.0)
```

**Stage 3 — Flavor Vector Similarity**

Independent of the text, each drink holds an 8-dimensional flavor profile vector. The user's flavor focus selections are mapped into a matching query vector. Cosine similarity is computed in this 8-dimensional flavor space, capturing taste alignment that the words in a description might miss entirely.

**The Scoring Formula**

```
Score = 0.35 × TextSimilarity + 0.65 × FlavorSimilarity
```

The 65% weight on flavor similarity was arrived at through empirical tuning. An equal split caused drinks with keyword-heavy descriptions to rank above better-tasting matches — which defeats the entire purpose of the system.

![Hybrid Recommend Source Code](technical.jpg)

---

## Module 2 — FusionNet Neural Recipe Generator

**File:** `fusion_model.py` · **Framework:** PyTorch

FusionNet is a custom Multi-Layer Perceptron trained to solve a multi-label classification problem. Given a fused representation of two drinks, it predicts which ingredients from the vocabulary belong in the synthesized recipe — and how confident it is about each one.

### Network Architecture

```
Input
  ├── TF-IDF reduced text embedding
  ├── One-hot category encodings (type, temperature, sugar, caffeine)
  └── 8-dimensional flavor vector

  [Dense → ReLU → Dropout]
  [Dense → ReLU → Dropout]
  [Dense → ReLU]

Output
  └── Sigmoid activation over 50-ingredient vocabulary
              │
         Threshold θ
              │
  Final cohesive ingredient list
```

**Why Sigmoid instead of Softmax**

Softmax forces the model to pick one winner. Sigmoid lets each ingredient be evaluated independently, which is correct here — a real drink can have ice, whole milk, and espresso all at once. Multi-label output requires multi-label activation.

**Why Thresholding matters**

Without a threshold, early FusionNet versions returned 15 to 20 ingredient lists that looked more like a grocery run than a café drink. Threshold tuning was validated against manually reviewed real-world recipes until output lists consistently fell in the 5 to 10 ingredient range.

**How the two parent drinks are fused**

Both drinks' input vectors are averaged before being passed to the network. This encodes shared flavor heritage — if both drinks are creamy and sweet, the synthesized recipe inherits that — while still allowing the model to learn novel combinations that neither parent drink contains on its own.

---

## Application Walkthrough

### Step 1 — Authentication and User Persistence

The entry point is a sign-in screen connected to MongoDB Atlas. Authenticated users get persistent favourites and taste profile tracking. Guest mode works too, but without any data persistence.

![Authentication Screen](hero.jpg)

---

### Step 2 — The Recommendation Interface

Users set hard filters at the top — sugar level, caffeine, temperature — and optionally add a flavour focus tag. The Recommend button triggers the full hybrid pipeline and returns ranked results with a complete explainability breakdown for each match.

![AI Recommender Results](ai_recommender.jpg)

![Flavor Radar Detail](ai_recommender-1.jpg)

Each result shows:

- **Overall match score** — the final weighted hybrid score
- **Text similarity** — how well the drink's description matched the query
- **Flavor similarity** — how closely the 8D flavor vectors aligned
- **Key ingredients** — what is actually in the drink
- **Radar chart** — a visual overlay of the drink's flavor fingerprint, so users can see exactly which dimensions drove the match

This is deliberate explainability, not decoration. A user who asked for something creamy should be able to see the creamy axis spike on the chart and understand why that drink was chosen.

---

### Step 3 — MongoDB Atlas Live Data Layer

Every time a user adds a favourite, a structured document is written to the MongoDB Atlas `favorites` collection in real time.

![MongoDB Atlas Favorites Collection](database_layer.jpg)

```json
{
  "_id": ObjectId("6981a0285d2f737d997cf11d"),
  "event_type": "favorite",
  "session_id": "1feb06b9-b7e0-44bd-9820-63bdf6fe0808",
  "drink_id": 1,
  "action": "add",
  "created_at": "2026-02-03T07:13:44.585+00:00"
}
```

This schema was designed with future capability in mind. The session-level event log is the foundation for re-ranking by personal history, taste drift detection over time, and eventually collaborative filtering across users.

---

### Step 4 — The Fusion Lab

Users pick two drinks as parents. FusionNet runs inference and returns a named fusion recipe with a full ingredient list, per-ingredient confidence scores, step-by-step preparation instructions, and a flavor radar for the synthesized drink.

![FusionNet Output — Ingredients and Confidence](fusion_drink.jpg)

![FusionNet Output — Preparation Steps and Radar](fusion_drink_2.jpg)

The confidence scores tell an interpretable story. In the example shown — a fusion of Hot Caramel and Hot Toasted Pecan Latte — Ice (64%) and Whole Milk (53%) score high because both parent drinks share a dairy-and-base structure. Brewed Coffee (29%) comes from the Caramel parent. Pumpkin Spice Syrup (12%) is the warm-spice signature of the Pecan Latte bleeding into the synthesis. Ingredients at or below the threshold are cut entirely.

---

## Engineering Challenges Worth Highlighting

**Weight calibration in the hybrid scorer**

Starting with equal weights (0.5/0.5) produced results where drinks with descriptive, keyword-rich names ranked above better flavor matches. The shift to 0.35/0.65 was not arbitrary — it came from reviewing specific failure cases where a drink called "Tropical Mango Paradise" would outscore a genuinely fruity drink simply because the words matched, not the taste.

**Multi-label threshold tuning**

The sigmoid output required careful threshold selection. Too low and the ingredient list became incoherent. Too high and the model would sometimes return nothing at all for unusual fusions. The final value was validated against a manually curated set of plausible café drinks rather than against any automated metric.

**Serverless MongoDB connectivity**

Hugging Face Spaces is a stateless serverless environment. MongoDB Atlas connections that work fine locally can fail or hang in that context due to SSL certificate handling and connection pool exhaustion under concurrent users. The solution involved explicit SSL configuration, short connection timeouts, and a graceful fallback to guest mode when the database is unreachable — so the recommendation engine always works even if persistence temporarily does not.

---

## Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Frontend | Streamlit | UI, radar plots, session state management |
| NLP | Scikit-Learn | TF-IDF vectorizer, cosine similarity |
| Deep Learning | PyTorch | FusionNet MLP, Sigmoid multi-label output |
| Database | MongoDB Atlas | Favourites persistence, session event logging |
| Deployment | Hugging Face Spaces | Serverless inference and UI hosting |
| Data layer | Pandas, NumPy | Dataset management, vector operations |
| Visualization | Plotly | 8-axis interactive radar charts |

---

## Local Setup

```bash
git clone https://github.com/your-username/flavorgen-cafe.git
cd flavorgen-cafe

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Add your MONGODB_URI to .env

streamlit run app.py
```

Your `.env` file needs one variable:

```
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/flavorgen-cafe
```

---

## Repository Structure

```
flavorgen-cafe/
├── app.py
├── semantic_model.py
├── fusion_model.py
├── data/
│   └── drinks_hybrid_augmented.csv
├── models/
│   └── fusionnet_weights.pt
├── ai_recommender.jpg
├── ai_recommender-1.jpg
├── database_layer.jpg
├── dataset_structure.jpg
├── fusion_drink.jpg
├── fusion_drink_2.jpg
├── hero.jpg
├── technical.jpg
├── requirements.txt
└── README.md
```

---

## What Comes Next

The MongoDB event log is already collecting the data needed for collaborative filtering — the next logical step is using actual user behaviour to re-rank recommendations rather than relying solely on flavor vectors. Beyond that, replacing TF-IDF with dense sentence embeddings would dramatically improve semantic matching for unusual or creative queries. FusionNet itself is a strong baseline but a Transformer-based architecture with ingredient-level attention would produce more interpretable and controllable synthesis outputs.

---

## Contact

**Lakshmi Sruthi Anchula**  
Applied Machine Learning Engineer · Full-Stack AI Systems Builder

Open to roles in ML Engineering, Recommender Systems, AI Product Development, and Full-Stack AI Applications.

[LinkedIn](https://your-linkedin-link) · [GitHub](https://your-github-link) · [Email](mailto:your-email@example.com)