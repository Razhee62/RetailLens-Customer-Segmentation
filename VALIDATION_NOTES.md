# RetailLens validation notebook — upload notes

The included `notebooks/RetailLens_Validation.ipynb` reproduces the published ROC-AUC and segment-validation metrics from two anonymized *historical feature checkpoints*.

## Upload to the existing GitHub repository

From the repository page choose **Add file → Upload files**, then drag the `notebooks` and `data` folders, together with `requirements.txt` and this file, into the upload area. Commit the changes.

**Do not upload the ZIP itself.** GitHub must see the extracted files.

## Run locally

1. Download or clone the repository.
2. Install libraries: `pip install -r requirements.txt`.
3. Start Jupyter from the repository root, open `notebooks/RetailLens_Validation.ipynb`, and run all cells.

In Google Colab, clone the repository first, or upload both CSVs into `data/` and ensure that `data/` is under the current working directory (or its parent).

## Scope

The two CSVs deliberately exclude raw transaction rows, Customer IDs and monetary totals. The notebook confirms **downstream validation only**. It is not the complete raw Excel data processing, cancellation matching or RFM feature engineering pipeline. The original 277-cell exploratory notebook should be preserved separately until it has been cleaned and independently rerun.

## Suggested addition to the main README

### Reproduce the validation results

The [`RetailLens_Validation.ipynb`](notebooks/RetailLens_Validation.ipynb) notebook recalculates the two-cutoff ROC-AUC comparisons, paired bootstrap intervals, segment repeat-purchase rates and September outcome sensitivity from the anonymized CSV checkpoints in `data/`. Start Jupyter in the repository root and run all cells. This notebook validates saved customer-level results; it does not regenerate them from the original Excel workbook.
