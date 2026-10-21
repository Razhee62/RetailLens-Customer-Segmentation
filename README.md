# RetailLens

### Customer segmentation and purchase behavior analysis using RFM

A transparent, evidence-driven analysis of **1,067,371 historical retail transaction records** from the [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii). RetailLens transforms transaction data into Recency, Frequency, and Monetary (RFM) customer profiles, builds eight interpretable customer segments, and checks how well historical purchase patterns distinguish customers who purchase again within 90 days.

**Stack:** Python · pandas · NumPy · matplotlib · scikit-learn · SciPy · statsmodels · Google Colab

**Status:** Historical analysis and two temporal validations completed. The original notebook is exploratory and still needs a clean-runtime reproducibility check. This is **not** a deployed model or a claim of increased business revenue.

## At a glance

| Measure | Result |
|---|---:|
| Original transaction records | 1,067,371 |
| Full-period customers after the documented adjustment | 5,843 |
| Customer segments | 8 |
| June 2011 combined RFM ROC-AUC | **0.8066** |
| September 2011 combined RFM ROC-AUC | **0.7817** |
| September 2011 eight-segment ROC-AUC, using June rates | **0.7684** |

## 1. Problem and objectives

Retailers often have transaction data but no simple way to turn that history into useful customer-level descriptions. This project asks:

> Do Recency, Frequency, and Monetary purchasing characteristics help distinguish which existing customers will make at least one qualifying purchase in the next 90 days?

The objectives were to:

1. Audit and clean two years of transaction data without silently removing meaningful records.
2. Construct customer-level RFM features from eligible product purchases.
3. Create eight understandable, rule-based customer groups.
4. Evaluate observed 90-day purchase activity using historical cutoffs rather than a random transaction split.
5. Compare a combined RFM score with a simple Recency-only baseline and examine uncertainty.
6. Test whether the main findings are sensitive to selected matched cancellations.

The project uses **interpretable scores and rules**, not a fitted machine-learning classifier.

## 2. Dataset

**Source:** Daqing Chen, *Online Retail II*, UCI Machine Learning Repository ([dataset page](https://archive.ics.uci.edu/dataset/502/online+retail+ii); [DOI](https://doi.org/10.24432/C5CG6D)).

- **Coverage:** December 1, 2009 to December 9, 2011.
- **Source workbook:** `online_retail_II.xlsx`, with two year-based worksheets.
- **Business context:** Transactions from a UK-based non-store retailer; the dataset includes wholesale purchasing activity.
- **Fields used:** Invoice, StockCode, Quantity, InvoiceDate, Price, Customer ID, and other supporting fields.

The official source is distributed under **CC BY 4.0**. Please credit UCI and the dataset creator when using or redistributing the data. The raw workbook is not included in this starter repository.

## 3. Data preparation and transaction policy

Data cleaning was treated as an analytical decision, not simply a way to reduce the row count.

| Audit / processing stage | Observed count |
|---|---:|
| Raw rows across both worksheets | 1,067,371 |
| Rows missing a customer ID | 243,007 |
| Additional exact duplicate rows in raw data | 34,335 |
| Eligible merchandise purchase lines after exact deduplication | 776,596 |
| Strict, one-to-one purchase–cancellation matches (30-day window) | 4,337 |
| Full-period purchase lines after removing matched reversals | 772,259 |
| Full-period identified customers after adjustment | 5,843 |
| Full-period distinct qualifying invoices after adjustment | 36,388 |

### Purchase eligibility

A qualifying purchase line required a known customer ID, positive quantity, and positive unit price. Selected service, charge, manual, test, and adjustment codes were excluded after inspection; some unusual-looking product codes were retained when the evidence supported treating them as merchandise. Exact duplicate rows were removed from the eligible purchase base.

**Why not simply delete all unusual records?** Some extreme purchases may be real, and not every cancellation can be identified from a negative quantity alone. The cleaning policy therefore emphasizes explicit rules and sensitivity checks.

### Conservative cancellation matching

Candidate cancellation lines were linked to an earlier, unused purchase line only when all of the following matched:

- Customer ID and normalized product code.
- Absolute quantity and unit price, with price represented at **0.001-unit precision** for matching.
- Purchase strictly before the cancellation, no more than **30 days** earlier.
- A purchase line could be matched only once; the most recent available eligible purchase was preferred.

This yielded **4,337 strict matches**. A previous exploratory check that allowed equal timestamps produced 4,343, but the **strictly earlier** timing rule is the final documented method. It is a conservative reversal adjustment, **not** complete refund accounting: partial returns, unmatched cancellations, or late returns may remain.

For the full-period descriptive RFM view, historical purchase value after matched-reversal removal was **16,602,190.40** in source price units; this is **not profit**, and should not be presented as audited net revenue or lifetime value.

## 4. RFM feature engineering

For each customer, measured at a particular historical cutoff:

| Feature | Definition | Favorable direction |
|---|---|---|
| **Recency (R)** | Calendar days since the latest qualifying historical purchase | Lower |
| **Frequency (F)** | Number of **distinct invoices**, not item rows | Higher |
| **Monetary (M)** | Sum of eligible historical `Quantity × Price` line values after known matched reversals | Higher |

**Why invoice count?** A single invoice can contain multiple items; counting product rows would mistakenly treat one shopping occasion as several separate purchases.

Each feature was converted into a **1–5 rank-based score** using average percentile ranks for ties and `ceil(percentile_rank × 5)`. Recency was reversed so a more recent customer receives a higher score. The combined RFM ranking score is:

```text
Combined RFM = R_score + F_score + M_score
```

It is a **predefined equal-weight score**, not an estimated probability and not a supervised model. Because of ties, the score groups are not guaranteed to contain exactly equal numbers of customers.

## 5. Eight customer segments

Segments were assigned by **ordered rules** on the three 1–5 scores. The first matching rule wins, so the conditions do not have to be mutually exclusive before priority is applied.

| Priority | Segment | Score rule |
|---:|---|---|
| 1 | Champions | R ≥ 4, F ≥ 4, M ≥ 4 |
| 2 | High-Value Lapsed | R ≤ 2, M ≥ 4 |
| 3 | At Risk | R ≤ 2, F ≥ 3 |
| 4 | Hibernating | R ≤ 2, F ≤ 2 |
| 5 | Loyal Customers | R ≥ 3, F ≥ 4 |
| 6 | Potential Loyalists | R ≥ 4, F = 2 or 3 |
| 7 | Recent One-Time Buyers | R ≥ 4, F = 1 |
| 8 | Needs Attention | R = 3, F ≤ 3 |

All **5,843** customers in the full-period adjusted view were assigned to one segment. The segment names are business-friendly labels, **not independent facts about individual customers**. In particular, “Recent One-Time Buyers” uses an **F score of 1**, which does not necessarily mean literally one invoice.

These categories are intended for explanation and to generate testable customer-management hypotheses, not to prove that any marketing intervention is effective.

## 6. Historical validation design

To avoid using future information when building customer features, each validation separates the **historical observation period** from a later **90-day outcome period**.

| | Earlier validation | Later validation |
|---|---|---|
| Feature cutoff | June 10, 2011 | September 10, 2011 |
| Outcome window | June 10–September 8, 2011 **(end exclusive)** | September 10–December 9, 2011 **(end exclusive)** |
| Identified historical customers | 4,938 | 5,251 |
| Customers with a qualifying purchase in the next 90 days | 1,581 | 2,286 |
| Observed 90-day repeat-purchase rate | 32.02% | 43.53% |

The binary target was `Repeat90d = 1` if a customer present before the cutoff had **at least one positive qualifying merchandise purchase line** during the outcome window; otherwise it was 0.

**Leakage prevention:** RFM scores and segment assignment used only purchase history before the cutoff. Purchase reversals were removed from historical features only when the cancellation itself was recorded before that cutoff. Later events were used only to evaluate the outcome or in separate sensitivity analysis.

This target reflects **observed qualifying purchase activity**, not necessarily orders retained after every possible refund.

## 7. Results

### 7.1 Combined RFM vs. Recency-only baseline

| Metric | June 2011 | September 2011 |
|---|---:|---:|
| Recency-only ROC-AUC | 0.7637 | 0.7497 |
| **Combined RFM ROC-AUC** | **0.8066** | **0.7817** |
| Absolute AUC improvement | +0.0429 | +0.0320 |
| Paired bootstrap 95% CI for improvement | +0.0322 to +0.0524 | +0.0228 to +0.0417 |

Bootstrap intervals used **500 paired customer-level resamples** per period with a fixed random seed. Both intervals exclude zero, supporting positive AUC differences for the evaluated cohorts under the bootstrap assumptions.

![Comparison of Recency-only and Combined RFM ROC-AUC in June and September 2011](figures/auc_comparison.png)

**Interpretation:** The combined score provided better **ranking discrimination** of subsequent purchasers in both periods. An ROC-AUC of 0.8066 **does not mean 80.66% classification accuracy**.

### 7.2 Segment behavior over time

| Segment | June customers | June repeat rate | September customers | September repeat rate |
|---|---:|---:|---:|---:|
| Champions | 1,136 | 71.13% | 1,220 | 80.25% |
| Loyal Customers | 565 | 39.29% | 628 | 55.41% |
| Potential Loyalists | 424 | 32.08% | 442 | 50.68% |
| Recent One-Time Buyers | 274 | 32.12% | 238 | 46.22% |
| High-Value Lapsed | 358 | 19.83% | 377 | 31.03% |
| Needs Attention | 564 | 18.97% | 619 | 28.11% |
| At Risk | 622 | 14.15% | 287 | 28.92% |
| Hibernating | 995 | 6.13% | 1,440 | 17.43% |

![Observed 90-day repeat-purchase rates for eight segments at both historical cutoffs](figures/segment_repeat_rates.png)

Champions had the highest observed rate in both periods; Hibernating customers had the lowest. However, rates changed across periods, so a segment label should **not** be presented as having one fixed repeat-purchase probability.

### 7.3 Testing the eight-segment approach separately

The **June segment-level repeat rates** were mapped onto the corresponding September segment labels. Using these earlier rates as ranking scores gave a **September ROC-AUC of 0.7684**.

| September comparison | ROC-AUC |
|---|---:|
| Recency-only | 0.7497 |
| Eight segments using June rates | 0.7684 |
| Combined RFM | 0.7817 |

Paired bootstrap comparisons for the September cohort (500 resamples):

- **Segments minus Recency:** +0.0187; 95% CI **+0.0105 to +0.0261**.
- **Segments minus Combined RFM:** −0.0133; 95% CI **−0.0193 to −0.0077**.

The segments provided useful interpretable ranking information, while the combined RFM score retained higher observed discrimination in this evaluation. These periods are from the **same retailer** and involve some of the **same customers**; they are not independent external validation.

### 7.4 Cancellation sensitivity

A sensitivity check excluded future purchase lines matched to cancellations **recorded before the outcome window ended**.

| Metric | June 2011 | September 2011 |
|---|---:|---:|
| Original 90-day repeat purchasers | 1,581 | 2,286 |
| Repeat purchasers after matched-cancellation adjustment | 1,580 | 2,284 |
| Future purchase lines removed | 365 | 788 |

The September Combined RFM AUC was **0.7817** both before and after this adjustment **when rounded to four decimal places**. This supports stability under the *specific* matching policy, not under every possible return or accounting scenario.

## 8. Repository contents and reproducibility

This GitHub starter package contains:

```text
RetailLens/
├── README.md
└── figures/
    ├── auc_comparison.png
    └── segment_repeat_rates.png
```

The project was developed as an **interactive Google Colab notebook**. The original exploratory notebook is **not included in this starter package** and has not yet been verified as a complete top-to-bottom, fresh-runtime execution. Before describing the repository as fully reproducible:

1. Add a cleaned notebook at `notebooks/retail_lens_analysis.ipynb`.
2. Download `online_retail_II.xlsx` from the [UCI dataset page](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and configure its path in the notebook.
3. Run the notebook from a fresh Colab runtime, verifying the counts and AUC outputs above.
4. Save the environment/dependency versions and document any deviation from the final eligibility and matching policy.
5. Keep raw customer-level exports out of the repository unless you have a clear need and have checked appropriate data-use considerations.

**Dependencies used during exploration:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`, `statsmodels`, and `openpyxl`. Exact pinned versions are not yet documented; a tested `requirements.txt` should be added after the clean rerun.

## 9. Limitations

- Results reflect **one historical retailer**, including wholesale customers; they are not a validated modern or cross-company prediction system.
- Exact duplicate removal and merchandise eligibility are policy choices that can affect reported values and customer profiles.
- Matching exact cancellations within 30 days does not handle all partial returns, delayed reversals, or unmatched records.
- The validation outcome measures **positive purchase activity**, not lifetime value, customer profitability, or an audited retained-sales figure.
- The RFM score uses preset equal weights, and segment cutoffs are rule-based; these were not tuned through a held-out optimization procedure.
- Both temporal evaluations share the same underlying source and some customers, limiting claims of generalization.
- Bootstrap confidence intervals quantify customer-resampling uncertainty under stated assumptions, not uncertainty across companies or economic conditions.
- The project did **not** test a randomized marketing intervention or measure causal retention/revenue uplift.
- The exploratory notebook requires cleanup and a documented fresh-runtime reproduction test before public release as an end-to-end reproducible pipeline.

## 10. Next steps

- Refactor the exploratory Colab work into an ordered, reproducible notebook with configuration and automated checks.
- Evaluate sensitivity to different cancellation windows, alternative score definitions, and additional historical cutoffs.
- Test on a separate dataset and compare against additional simple scoring or supervised baselines with careful temporal design.
- Assess calibration and operational business value if probabilities, campaign decisions, and appropriate outcome/cost data become available.
- Evaluate proposed customer-management interventions with a controlled experiment before making uplift claims.

## Dataset attribution

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D). Source data licensed **CC BY 4.0**.

---

*RetailLens is a learning and research portfolio project. Feedback on methodological choices, validation, and reproducibility is welcome.*
