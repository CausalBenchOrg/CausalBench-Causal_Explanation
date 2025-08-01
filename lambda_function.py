from collections import defaultdict
from causal_analysis_helper import run_causal_analysis
import math
from causal_recommendation_helper import run_causal_recommendation
from helper_services.pdf_helper import generate_pdf
from mail_services import send_email
import numpy as np
import common_constants

def handler(event, context):
    causal_analysis_results, download_dir = run_causal_analysis(zip_urls=event.get('zip_urls', []),
                                    outcome_column=event.get('outcome_column', 'Time.Duration'),
                                    candidate_hyperparameters= event.get('candidate_hyperparameters', None))
    
    max_points = max(math.ceil(np.sqrt(len(event.get('zip_urls', [])))), 10)
    
    dimensions = defaultdict(dict)
    causal_recommendation_results = None
    # find all causal effects
    if not causal_analysis_results['grouped']:
        causal_effects = causal_analysis_results["causal_effects"]["effects"]
        for k, v in causal_effects.items():
            k = k.split(".")[1]  # Remove 'HP.' prefix
            if k in list(event.get('hyperparameter_limits', {}).keys()) and v != 0:
                dimensions[k]['strength'] = v
                dimensions[k]['min_val'] = event.get('hyperparameter_limits', {})[k]['min']
                dimensions[k]['max_val'] = event.get('hyperparameter_limits', {})[k]['max']
    else:
        causal_effects = causal_analysis_results["causal_effects"]["overall_effects"]
        for k, v in causal_effects.items():
            if k in list(event.get('hyperparameter_limits', {}).keys()) and v != 0:
                dimensions[k]['strength'] = v
                dimensions[k]['min_val'] = event.get('hyperparameter_limits', {})[k]['min']
                dimensions[k]['max_val'] = event.get('hyperparameter_limits', {})[k]['max']
    try:
        if len(dimensions) > 0:
            causal_recommendation_results = run_causal_recommendation(dimensions, max_points)
        else:
            print("Skipping Causal Recommendation as len(dimensions) == 0.")
    except Exception as e:
        print(f"Error during causal recommendation: {e}")
        print(f"Error during causal recommendation: {e}")
    finally:
        print(f"Causal Recommendation {causal_recommendation_results}!")

    pdf_filename = generate_pdf(causal_analysis_results, causal_recommendation_results, [f'HP.{var}' for var in list(dimensions.keys())], event.get('unique_id'), event.get('run_ids'), event.get('filters'))
    # attachments = [pdf_filename, analysis_logger_file_name]
    attachments = [pdf_filename]
    
    send_email(event.get('user_email'), "[CausalBench] Causal Analysis Results", common_constants.CAUSAL_ANALYSIS_EMAIL_BODY, attachments=attachments)
    
    # # remove the attachments after sending the email
    # for attachment in attachments:
    #     if os.path.exists(attachment):
    #         os.remove(attachment)
    
    response = {
        'analysis_results': causal_analysis_results,
        'causal_recommendation_results': causal_recommendation_results,
    }
    return response