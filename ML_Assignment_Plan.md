# ML Assignment — Plan & Dataset Options

_Fundamentals of Machine Learning — Programming Assignment (group of 2, 40 marks)._
_Working summary of decisions, dataset options, and the build plan._

---

## 1. What the assignment requires

Apply machine learning (**classical ML only — no deep learning**) to a real,
publicly-hosted dataset and write it up.

You choose **one** learning paradigm, and the number of algorithms depends on it:

| Paradigm | Algorithms required |
|----------|--------------------|
| Supervised | **two** distinct algorithms |
| Unsupervised / clustering | **two** distinct algorithms |
| Reinforcement learning | **one** algorithm |

> **Open question to confirm with the lecturer:** whether you pick **one** paradigm
> (my strong reading of the brief's "supervised, unsupervised _or_ reinforcement")
> or must apply all three. The scope of the whole project depends on this, and a
> missing/incorrect component means zero marks, so it is worth a one-line email:
> _"Do we pick one learning paradigm, or apply all three?"_
> Supervised is almost certainly part of the answer either way, so building it now
> is safe.

### Required deliverables (all mandatory — any missing = zero marks)
- `members.txt` — IDs and emails of both members
- `submission.txt` — dataset link + GitHub repo link + YouTube demo link
- **GitHub repo** with detailed commit history **from the start** (they check this)
- **YouTube video** — each member ≤5 min explaining their contribution
- **PDF report** — the main marked deliverable (problem, dataset, methodology,
  results, discussion), comparing the algorithms' accuracy with appropriate metrics
- **Source code as text in an appendix** of the report (not screenshots)
- Whole bundle zipped as `ML-assignment.zip`, uploaded by one member

### Watch-outs
- Turnitin similarity **> 20%** is penalised → all writing must be original.
- Resubmissions capped at **45%**.
- Dataset must **not** be a tutorial dataset and must be reasonably complex.

---

## 2. What makes a dataset score well here

- **Not a tutorial dataset** — avoid Iris, Titanic, Wine, Breast-Cancer-Wisconsin, Boston Housing.
- **Complex enough** — thousands of rows, 15+ mixed-type features.
- **Class imbalance** — lets you argue why accuracy alone is misleading and bring
  in precision, recall, F1, ROC-AUC, PR-AUC (the "appropriate metrics" the brief wants).
- **Some genuine cleaning needed** — undocumented codes, encoding, scaling.
- **Two algorithms that disagree** — gives a real compare-and-contrast section.

---

## 3. Recommended dataset options

All are real, publicly hosted, non-tutorial, CC BY 4.0, and load via `pip install ucimlrepo`.

### Finance
- **Default of Credit Card Clients** — UCI ID 350 · <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
  30,000 clients, 23 features, ~22% default. Real cleaning quirks. **← chosen (top pick).**
- **Bank Marketing** — UCI ID 222 · <https://archive.ics.uci.edu/dataset/222/bank+marketing>
  Up to 45,211 records, 16–20 features, predict term-deposit subscription. ~11% "yes" (strong imbalance).
- **Statlog German Credit** — UCI ID 144 · credit risk, 1,000 rows, 20 features. Has a cost matrix. Small but classic.

### Marketing / e-commerce
- **Online Shoppers Purchasing Intention** — UCI ID 468 · <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>
  12,330 sessions, 18 features, ~15% purchase rate. Clean but needs encoding + scaling.

### Health
- **Diabetes 130-US Hospitals (1999–2008)** — UCI ID 296 · <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>
  ~101,766 encounters, ~50 columns, predict 30-day readmission. Most complex option.
  _Caveat:_ very heavily used by students → higher Turnitin / originality risk; SVM slow on 100k rows.

### Multiclass / high-dimensional (good for showing PCA / dimensionality reduction)
- **Human Activity Recognition Using Smartphones** — UCI ID 240 · ~10,299 rows, **561 features**, 6 classes. Ideal PCA showcase.
- **Predict Students' Dropout & Academic Success** — UCI ID 697 · ~4,424 rows, 36 features, 3 classes.

### Quick chooser
- Cleanest full-marks path → **Credit Card Default** or **Bank Marketing**
- Maximise preprocessing marks → **Diabetes 130** (mind Turnitin) or **HAR** (PCA)
- Most original / less picked-over → **Bank Marketing** or **Online Shoppers**

---

## 4. Chosen dataset — Default of Credit Card Clients

- **Source:** UCI Machine Learning Repository, ID 350.
- **Citation:** Yeh, I. (2009). _Default of Credit Card Clients_. UCI Machine Learning Repository. <https://doi.org/10.24432/C55S3H> (CC BY 4.0).
- **Task:** binary classification — will a Taiwanese bank client **default next month** (Oct 2005)?
- **Size:** 30,000 rows × 23 features + 1 target. **~22% default / 78% no-default** (imbalanced). No missing values.

### Feature groups
| Group | Features |
|-------|----------|
| Demographics | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Repayment status | `PAY_1`…`PAY_6` (−1 = paid on time, 1–9 = months late, special codes −2/0) |
| Bill amounts | `BILL_AMT1`…`BILL_AMT6` |
| Payment amounts | `PAY_AMT1`…`PAY_AMT6` |
| **Target** | `default.payment.next.month` (1 = default, 0 = no) |

### Known cleaning quirks (worth marks)
- `EDUCATION` has undocumented codes 0, 5, 6 → fold into 4 ("others").
- `MARRIAGE` has undocumented code 0 → fold into 3 ("others").
- Drop the `ID` column (no predictive value).

---

## 5. Detailed plan

### Learning paradigm & algorithms
- **Supervised — binary classification.**
- **Two distinct algorithms, one per member:**
  - **Member A → Logistic Regression** (interpretable linear baseline; scaled features)
  - **Member B → Random Forest** (non-linear tree ensemble; scale-invariant)
- Both use `class_weight='balanced'` to handle the 78/22 imbalance.

### Preprocessing
1. Drop `ID`.
2. Fold undocumented `EDUCATION` / `MARRIAGE` codes into "others".
3. One-hot encode nominal features (`SEX`, `EDUCATION`, `MARRIAGE`); keep `PAY_*` as ordinal numeric.
4. Standard-scale numeric features for Logistic Regression (Random Forest gets unscaled).
5. Stratified 80/20 train/test split (preserves default rate).

### Evaluation metrics (the compare-and-contrast core)
- Accuracy, **Precision, Recall, F1**, **ROC-AUC**, **PR-AUC**
- Confusion matrices, ROC & PR curves
- 5-fold stratified cross-validation (ROC-AUC)
- Narrative: **why accuracy alone misleads** on imbalanced data.

### Results already computed (real, from the actual data)
| Metric | Logistic Regression | Random Forest |
|--------|--------------------:|--------------:|
| Accuracy | 0.679 | **0.803** |
| Precision | 0.368 | **0.560** |
| Recall | **0.631** | 0.516 |
| F1 | 0.465 | **0.537** |
| ROC-AUC | 0.710 | **0.771** |
| PR-AUC | 0.491 | **0.551** |
| CV ROC-AUC | 0.728 ± 0.011 | **0.781 ± 0.004** |

**Interpretation:** Random Forest is the stronger overall ranker (higher AUC/PR-AUC);
Logistic Regression (balanced) catches more true defaulters (higher recall) but with
many more false alarms → the "better" model depends on the bank's cost trade-off.

### Discussion / future work (report section)
- Threshold tuning on the PR curve to a business recall target.
- Hyper-parameter search (`GridSearchCV` / `RandomizedSearchCV`).
- SMOTE / undersampling as an alternative to class weights.
- Feature engineering (credit utilisation `BILL_AMT/LIMIT_BAL`, delinquency counts).
- Try Gradient Boosting as a stronger classical model.
- Limitations: single bank / 2005 / Taiwan; ambiguous repayment codes; fairness of demographic features.

---

## 6. Deliverables split

### What I (Claude) am producing
1. **Jupyter notebook** — clean, fully commented, real embedded outputs. _(built)_
2. **PDF report** — mapped to every rubric line, with source code as a text appendix. _(next)_
3. **`members.txt`** and **`submission.txt`** templates.

### What only the team can do
- YouTube demo video (≤5 min each).
- The real GitHub commit history — **build it up gradually from the start**, they verify it.
- The viva — be ready to defend every choice.
- Fill in names, IDs, emails, and each member's individual-contribution statement.

---

## 7. Immediate next steps
1. **Confirm scope with the lecturer** (one paradigm vs. all three).
2. Finish + run the notebook (real outputs). _(in progress)_
3. Generate the PDF report with figures + code appendix.
4. Create the GitHub repo early and commit in small, meaningful steps.
5. Record the demo video; assemble the `ML-assignment.zip` bundle.
