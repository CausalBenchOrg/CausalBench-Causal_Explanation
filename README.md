# CausalBench — Causal Explanation Lambda

This repository provides the AWS Lambda implementation for **CausalBench**, a framework for causal analysis and recommendation.  
It includes helper modules for causal scoring/explanations, conversion of YAML configs to CSV, and email delivery of results.

---

## Repository Structure

```
.
├── lambda_function.py           # Lambda entrypoint (AWS handler)
├── causal_analysis_helper.py    # Causal analysis utilities
├── causal_recommendation_helper.py
├── common_constants.py          # Configure env vars here (email ID, password)
├── mail_services.py             # SMTP email sender
├── mail_helper_services.py      # Email formatting and helpers
├── yaml_to_csv.py               # Convert YAML files downloaded from causalbench.org to CSV
├── media/                       # Static assets
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container build for AWS Lambda
└── LICENSE                      # Apache 2.0
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

### Convert YAML → CSV
```bash
python yaml_to_csv.py config.yaml --out out.csv
```

### Run helpers locally
```bash
python causal_analysis_helper.py --input data.csv
python causal_recommendation_helper.py --input data.csv
```

### Send a test email
```bash
python mail_services.py
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
> `lambda_function.lambda_handler`

## License

This project is licensed under the [Apache License 2.0](LICENSE).
