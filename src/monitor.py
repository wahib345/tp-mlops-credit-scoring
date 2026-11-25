import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def run_drift_analysis(
    reference_path: str = "data/credit_scoring.csv",
    current_path: str = "data/production_requests.csv",
    output_html: str = "reports/data_drift_report.html"
):
    ref_df = pd.read_csv(reference_path).drop(columns=["default_risk"], errors="ignore")
    cur_df = pd.read_csv(current_path).drop(columns=["default_risk"], errors="ignore")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    report.save_html(output_html)
    print(f"Rapport de Data Drift généré avec succès : {output_html}")


if __name__ == "__main__":
    run_drift_analysis()
