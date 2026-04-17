# CausalBench - Causal Explanation Lambda

This repository provides the AWS Lambda implementation for **CausalBench**, a framework for causal analysis and recommendation.
It includes helper modules for causal scoring/explanations, conversion of YAML configs to CSV, and email delivery of results.

---

## Repository Structure

```text
.
├── lambda_function.py                           # Lambda entrypoint (AWS handler)
├── event.json                                   # Sample local invocation payload
├── test_invoke.py                               # Local Lambda invocation helper
├── docker_commands.sh                           # Docker utility commands
├── common/
│   ├── common_constants.py                      # Shared constants/config
│   └── yaml_to_csv.py                           # Convert YAML files to CSV
├── helper_services/
│   ├── causal_analysis_helper.py                # Causal analysis utilities
│   ├── causal_recommendation_helper.py          # Legacy recommendation helper
│   ├── g2s_causal_recommendation_helper.py      # Current G2S recommendation helper
│   ├── download_helper.py
│   ├── hp_dtype_helper.py
│   ├── mail_helper.py                           # SMTP email sender
│   └── report_helper.py
├── images/                                      # Static assets
├── requirements.txt                             # Python dependencies
├── Dockerfile                                   # Container build for AWS Lambda
└── LICENSE                                      # Apache 2.0
```

---

## Installation (local)

1. **Clone and create a virtual environment**
   ```bash
   git clone https://github.com/CausalBenchOrg/CausalBench-Causal_Explanation.git
   cd CausalBench-Causal_Explanation
   python3 -m venv .venv && source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Set these in your shell (or via AWS Parameter Store/Secrets Manager):

   ```bash
   export SEND_AS_EMAIL="<your_sending_email>"
   export EMAIL="<your_email>"
   export EMAIL_PASSWORD="<your_app_password>"
   export SMTP_HOST="smtp.example.com"
   export SMTP_PORT="587"
   ```

---

## Usage

### Convert YAML to CSV
```bash
python common/yaml_to_csv.py config.yaml --out out.csv
```

### Invoke locally
```bash
python test_invoke.py
```

---

## AWS Lambda Deployment

This project supports deployment as a **container image**.

1. **Build the image**
   ```bash
   docker build -t causal-expl-lambda .
   ```

2. **(Optional) Test locally**
   ```bash
   docker run --rm -p 9000:8080 causal-expl-lambda
   curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
        -d '{"action":"analyze","payload":{"dataset":"s3://.../input.csv"}}'
   ```

3. **Deploy to AWS**
   - Push the image to Amazon ECR.
   - Create a Lambda function with image type.
   - Configure environment variables in the Lambda console.

> If deploying as a zip instead, set handler to:
> `lambda_function.handler`

## License

This project is licensed under the [Apache License 2.0](LICENSE).
