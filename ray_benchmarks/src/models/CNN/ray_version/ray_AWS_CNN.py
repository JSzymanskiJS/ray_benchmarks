from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://127.0.0.1:8265")

kick_off_ray_AWS_CNN_benchmark = (
    # Clone ray. If ray is already present, don't clone again.
    "git clone https://github.com/JSzymanskiJS/ray_benchmarks.git || true;"
    "conda create --name ray_benchmarks python=3.10 -y"
    "conda activate ray_benchmarks"
    "conda install jupyter -y"
    "pip install torch torchvision torchaudio"
    "pip install numpy pandas boto3 tqdm tabulate"
    'pip install -U "ray[default]"'
    # Run the benchmark.
    " python ray_benchmarks/src/models/CNN/ray_version/ray_AWS_CNN.py"
)


submission_id = client.submit_job(
    entrypoint=kick_off_ray_AWS_CNN_benchmark,
)

print("Use the following command to follow this Job's logs:")
print(f"ray job logs '{submission_id}' --follow")