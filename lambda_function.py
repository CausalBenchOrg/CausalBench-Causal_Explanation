from collections import defaultdict
import yaml
from helper_services.causal_analysis_helper import run_causal_analysis
import math
from helper_services.causal_recommendation_helper import run_causal_recommendation
from helper_services.download_helper import download_files
from helper_services.pdf_helper import generate_pdf
from helper_services.hp_dtype_helper import get_hp_dtypes
from helper_services.mail_helper import send_email
import numpy as np
from common.common_constants import CAUSAL_ANALYSIS_EMAIL_BODY, TEMP_DIR


def handler(event, context):
    # maximum recommended points
    max_points = max(math.ceil(np.sqrt(len(event.get('zip_urls', [])))), 10)

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
                data = list(group_data["data"][cols].itertuples(index=False, name=None))
                group_data['recommendations'] = run_causal_recommendation(data, dimensions, hp_dtypes, max_points)[0]
            else:
                print(f"Skipping Causal Recommendation for {group} as len(dimensions) == 0.")
        except Exception as e:
            print(f"Error during causal recommendation: {e}")
        finally:
            print(f"Causal Recommendation {group_data['recommendations']}!")

        group_data['data'] = len(group_data['data'])

    # save to file
    output_filename=f'{TEMP_DIR}/causal_analysis_results.yaml'
    with open(output_filename, 'w') as yaml_file:
        yaml.dump(causal_analysis_results, yaml_file, default_flow_style=False, indent=4, sort_keys=False)

    print(f"Results saved to {output_filename}")
    
    pdf_filename = generate_pdf(outcome_column, causal_analysis_results, event.get('unique_id'), event.get('run_ids'), event.get('filters'))

    # attachments = [pdf_filename, analysis_logger_file_name]
    attachments = [pdf_filename]
    
    try:
        send_email(event.get('user_email'), "[CausalBench] Causal Analysis Results", CAUSAL_ANALYSIS_EMAIL_BODY, attachments=attachments)
    except Exception as e:
        print(f"Error sending email: {e}")
    
    # # remove the attachments after sending the email
    # for attachment in attachments:
    #     if os.path.exists(attachment):
    #         os.remove(attachment)
    
    response = {
        "analysis_results": causal_analysis_results
    }
    
    return response
