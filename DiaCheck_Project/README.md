# DiaCheck — educational diabetes risk-factor ML project

This beginner-friendly project includes a Streamlit website, a Logistic Regression training/evaluation script, and downloadable PDF report generation.

## Medical safety and limitations

**Coursework prototype only — not a medical device or diagnostic tool.** Never use the score to diagnose, rule out, or treat diabetes. The displayed score is not a personal medical probability. Consult a qualified healthcare professional for health concerns.

The expected dataset is the **Diabetes Health Indicators Dataset**, derived from the US CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey. It uses self-reported survey indicators; the target is survey-based and is not necessarily a diagnosis confirmed by clinical records. The dataset is US-focused and may not generalize to India or to any individual. This model has not been externally or clinically validated.

This version uses general health/lifestyle factors only. It does not request lab values or symptoms.

## Dataset setup

Dataset page: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset  
CDC BRFSS background: https://www.cdc.gov/brfss/

Read the dataset description and terms. Download `diabetes_binary_health_indicators_BRFSS2015.csv` and put it in the project's `data` folder:

`data/diabetes_binary_health_indicators_BRFSS2015.csv`

Do not rename the columns.

## Run in VS Code on Windows

1. Extract the ZIP file.
2. In VS Code, select **File → Open Folder** and open `DiaCheck_Project`.
3. Select **Terminal → New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
5. Activate it in PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If activation is blocked, select a Command Prompt terminal and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
6. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
7. Download the CSV above and place it in `data/`.
8. Train and evaluate the model:
   ```powershell
   python train_model.py
   ```
9. Launch the website:
   ```powershell
   python -m streamlit run app.py
   ```
10. The app opens in your browser. Enter demo values, click **Estimate educational risk pattern**, then **Download PDF report**.

## How evaluation works

The script uses a stratified 80/20 train/test split and prints balanced accuracy, positive-class recall/sensitivity, precision, specificity, ROC AUC, classification report, and confusion matrix. Review sensitivity and specificity as well as accuracy. False negatives matter in screening contexts. These are only internal test-set metrics, not clinical validation.

## Limitations and responsible next steps

- Self-reported data may contain errors.
- The dataset is US-focused; performance may differ by country, age, and demographic group.
- A random held-out split is not external validation.
- The score is not guaranteed to be calibrated as an actual probability.
- Survey target categories do not necessarily equal a clinician-confirmed diagnosis.
- Real-world use would require expert review, representative external validation, calibration, fairness analysis, privacy/security review, and clinical/regulatory oversight.
- Do not enter identifiable personal information. This app does not ask for names or store form submissions in a database.

## Files

- `app.py`: Streamlit UI and PDF generation
- `train_model.py`: dataset loading, model training, evaluation, saving
- `requirements.txt`: dependencies
- `data/`: place the dataset here
- `models/`: model artifacts are saved here after training
