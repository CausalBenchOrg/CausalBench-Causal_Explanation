from collections import defaultdict
import os
from tempfile import tempdir
import tempfile

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
from common.common_constants import CAUSAL_ANALYSIS_EMAIL_BODY



def configure_env():
    """
    Directory setup to ensure isolation
    """
    # fake temporary directory
    temp_dir = tempfile.mkdtemp()
    tempfile.tempdir = None
    os.environ["TMPDIR"] = temp_dir
    os.environ["TEMP"] = temp_dir
    os.environ["TMP"] = temp_dir

    # fake home directory
    home_dir = os.path.join(temp_dir, "home")
    os.makedirs(home_dir, exist_ok=True)
    os.environ["HOME"] = home_dir
    os.environ["USERPROFILE"] = home_dir

    # fake mpl config directory
    os.environ["MPLCONFIGDIR"] = os.path.join(temp_dir, "mplconfig")


def handler(event, context):
    # configure the environment variables
    configure_env()

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
        send_email(event.get('user_email'), "[CausalBench] Causal Analysis Results", CAUSAL_ANALYSIS_EMAIL_BODY, attachments=attachments)
    except Exception as e:
        print(f"Error sending email: {e}")
    
    response = {
        "analysis_results": causal_analysis_results
    }
    
    return response
