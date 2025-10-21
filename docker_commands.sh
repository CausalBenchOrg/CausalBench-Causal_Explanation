aws ecr get-login-password --region us-west-2 | \
docker login --username AWS --password-stdin 851055393544.dkr.ecr.us-west-2.amazonaws.com
docker buildx build --platform linux/amd64 --provenance=false -t causalbench-causal_analysis:test . --progress=plain 
docker tag causalbench-causal_analysis:test 851055393544.dkr.ecr.us-west-2.amazonaws.com/causalbench/causal_explanation:latest
docker push 851055393544.dkr.ecr.us-west-2.amazonaws.com/causalbench/causal_explanation:latest
# 851055393544.dkr.ecr.us-west-2.amazonaws.com/causalbench/causal_explanation@sha256:f250482b6d64ced7169f2a13604000325482661e1b46ece1783367b5845a639f