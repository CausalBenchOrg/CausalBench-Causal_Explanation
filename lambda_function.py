import atexit
from collections import defaultdict
import os

import causalbench
from helper_services.causal_analysis_helper import run_causal_analysis
import math
from helper_services.causal_recommendation_helper import run_causal_recommendation
from helper_services.g2s_causal_recommendation_helper import run_g2s_causal_recommendation
from helper_services.download_helper import download_files
from helper_services.report_helper import generate_report
from helper_services.hp_dtype_helper import get_hp_dtypes
from helper_services.mail_helper import send_email
import numpy as np
from common.common_constants import TEMP_DIR


def build_email_body(causal_analysis_results, event):
    outcome_column = event.get('outcome_column', 'Time.Duration')
    filters = event.get('filters', None)
    metadata = causal_analysis_results.get('_metadata', {})

    experiment_count = metadata.get('experiment_count', 0)
    insufficient_data = metadata.get('insufficient_data', False)
    insufficient_data_reason = metadata.get('insufficient_data_reason', None)

    lines = ["CausalBench+ Causal Analysis Report", ""]
    lines.append(f"Outcome metric: {outcome_column}")
    lines.append(f"Experiments: Effects on {outcome_column} ({experiment_count} experiments)")

    if filters:
        filter_str = ", ".join(f"{k}={v}" for k, v in filters.items()) if isinstance(filters, dict) else str(filters)
        lines.append(f"Filters applied: {filter_str}")

    lines.append("")

    if insufficient_data:
        lines.append("INSUFFICIENT DATA: Causal effects could not be computed.")
        if insufficient_data_reason:
            lines.append(f"Reason: {insufficient_data_reason}")
        lines.append("")
        lines.append("To get results, run more experiments with varied hyperparameter configurations.")
        lines.append("Minimum requirements: ≥3 data points per variable, ≥2 unique values per hyperparameter.")
    else:
        all_effects = {}
        for group, group_data in causal_analysis_results.items():
            if group == "_metadata":
                continue
            for k, v in group_data.get("effects", {}).items():
                if not (isinstance(v, float) and math.isnan(v)):
                    all_effects[k] = v

        if all_effects:
            sorted_effects = sorted(all_effects.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            lines.append("Top causal effects:")
            for hp, effect in sorted_effects:
                hp_name = hp.split(".", 1)[1] if "." in hp else hp
                sign = "+" if effect >= 0 else ""
                lines.append(f"  {hp_name}: {sign}{effect:.4f}")

    lines.append("")
    lines.append("Full results are in the attached PDF report.")

    return "\n".join(lines)


def handler(event, context):
    # create fake home to ensure isolation
    fake_home = os.path.abspath(os.path.join(TEMP_DIR, "home"))
    os.makedirs(fake_home, exist_ok=True)
    os.environ["HOME"] = fake_home
    os.environ["USERPROFILE"] = fake_home

    # set JWT token
    causalbench.services.auth.__access_token = event.get('jwt_token', None)

    # maximum recommended points
    max_points = max(math.ceil(np.sqrt(len(event.get('zip_urls', [])))), 50)

    # outcome column
    outcome_column = event.get('outcome_column', 'Time.Duration')

    # download zip files
    download_dir, downloaded_files = download_files(zip_urls=event.get('zip_urls', []))

    # find all hyperparameter data types
    hp_dtypes = get_hp_dtypes(download_dir)
    
    # find all causal effects
    causal_analysis_results, download_dir = run_causal_analysis(
        download_dir=download_dir,
        hp_dtypes=hp_dtypes,
        outcome_column=outcome_column,
        candidate_hyperparameters= event.get('candidate_hyperparameters', None)
    )

    # find all causal recommendations
    for group, group_data in causal_analysis_results.items():
        if group == "_metadata":
            continue
        effects = group_data["effects"]
        dimensions = defaultdict(dict)
        for k, v in effects.items():
            k = k.split(".")[1]  # Remove 'HP.' prefix
            if k in list(event.get('hyperparameter_limits', {}).keys()) and v != 0:
                dimensions[k]['strength'] = v
                dimensions[k]['min_val'] = event.get('hyperparameter_limits', {})[k]['min']
                dimensions[k]['max_val'] = event.get('hyperparameter_limits', {})[k]['max']

        group_data['recommend_dims'] = [f'{var}' for var in list(dimensions.keys())]

        try:
            if len(dimensions) > 0:
                cols = ["HP." + dim for dim in dimensions.keys()]
                # data = list(group_data["data"][cols].itertuples(index=False, name=None))
                # group_data['recommendations'] = run_causal_recommendation(data, dimensions, hp_dtypes, max_points)
                sample_frame = group_data["data"][cols + ["outcome"]].copy()
                group_data['recommendations'] = run_g2s_causal_recommendation(sample_frame, dimensions, hp_dtypes, max_points)
            else:
                print(f"Skipping Causal Recommendation for {group} as len(dimensions) == 0.")
        except Exception as e:
            print(f"Error during causal recommendation: {e}")
        finally:
            print(f"Causal Recommendation {group_data['recommendations']}!")

        del group_data['data']
    
    yaml_filepath, pdf_filepath, xlsx_filepath = generate_report(outcome_column, causal_analysis_results, event.get('unique_id'), event.get('run_ids'), event.get('filters'))

    attachments = [pdf_filepath]
    if os.path.exists(xlsx_filepath):
        attachments.append(xlsx_filepath)
    
    try:
        send_email(event.get('user_email'), "[CausalBench] Causal Analysis Results", build_email_body(causal_analysis_results, event), attachments=attachments)
    except Exception as e:
        print(f"Error sending email: {e}")
    
    # # remove the attachments after sending the email
    # for attachment in attachments:
    #     if os.path.exists(attachment):
    #         atexit.register(lambda path=attachment: os.remove(path))
    
    response = {
        "analysis_results": causal_analysis_results
    }
    
    return response
