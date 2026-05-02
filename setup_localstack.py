"""
setup_localstack.py
────────────────────
Bootstraps LocalStack with:
  • IAM role  (knn-lambda-role)
  • IAM policy (S3 read/write on kickoff-analytics)
  • Lambda function (knn-predictor) from a local zip
  • Optionally invokes the function and prints results

Run:
    python setup_localstack.py [--invoke] [--zip-only]
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile

import boto3
from botocore.exceptions import ClientError

# ── Configuration ─────────────────────────────────────────────────────────────
ENDPOINT        = "http://localhost:4566"
REGION          = "us-east-1"
AWS_KEY         = "test"
AWS_SECRET      = "test"

BUCKET          = "kickoff-analytics"
FUNCTION_NAME   = "knn-predictor"
ROLE_NAME       = "knn-lambda-role"
POLICY_NAME     = "knn-s3-policy"
HANDLER         = "lambda_knn_handler.handler"
RUNTIME         = "python3.11"
TIMEOUT         = 900          # 15 min – Lambda max
MEMORY_MB       = 3008         # max Lambda RAM
LAMBDA_ZIP      = "lambda_knn.zip"
HANDLER_FILE    = "lambda_knn_handler.py"   # must match the file you deploy


# ── Boto3 factory ─────────────────────────────────────────────────────────────
def client(service):
    return boto3.client(
        service,
        endpoint_url=ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=REGION,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def print_step(msg):
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


def safe_create(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), ignore AlreadyExists / EntityAlreadyExists."""
    try:
        return fn(*args, **kwargs)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("EntityAlreadyExists", "ResourceConflictException",
                    "BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"    Already exists – skipping.")
            return None
        raise


# ── 1. S3 bucket ──────────────────────────────────────────────────────────────
def ensure_bucket():
    print_step(f"Ensuring S3 bucket: {BUCKET}")
    s3 = client("s3")
    safe_create(s3.create_bucket, Bucket=BUCKET)
    # Create results/ prefix placeholder
    s3.put_object(Bucket=BUCKET, Key="results/.keep", Body=b"")
    print(f"    s3://{BUCKET}/ ready")


# ── 2. IAM role & policy ──────────────────────────────────────────────────────
ASSUME_ROLE_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
})

S3_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": [
                f"arn:aws:s3:::{BUCKET}",
                f"arn:aws:s3:::{BUCKET}/*",
            ],
        },
        {
            "Effect": "Allow",
            "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
            "Resource": "*",
        },
    ],
})


def setup_iam() -> str:
    print_step("Setting up IAM role & policy")
    iam = client("iam")

    # Role
    role_res = safe_create(
        iam.create_role,
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=ASSUME_ROLE_POLICY,
        Description="Role for KNN Lambda predictor",
    )
    if role_res:
        role_arn = role_res["Role"]["Arn"]
        print(f"    Role created: {role_arn}")
    else:
        role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        print(f"    Role exists: {role_arn}")

    # Policy
    account_id = "000000000000"   # LocalStack default
    policy_arn = f"arn:aws:iam::{account_id}:policy/{POLICY_NAME}"
    safe_create(
        iam.create_policy,
        PolicyName=POLICY_NAME,
        PolicyDocument=S3_POLICY,
        Description="Allow Lambda to read/write kickoff-analytics S3 bucket",
    )
    # Attach policy to role (idempotent)
    try:
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy_arn)
        print(f"    Policy attached: {policy_arn}")
    except ClientError as e:
        if "already attached" not in str(e).lower():
            raise

    return role_arn


# ── 3. Zip Lambda code + dependencies ────────────────────────────────────────
# Minimal deps for Lambda — boto3/botocore are pre-installed in the
# Lambda runtime so we skip them here to save space.
# scikit-learn brings in numpy/scipy automatically.
DEPS = [
    "numpy",
    "pandas",
    "scikit-learn",
]

# Folders to exclude — intentionally NOT excluding numpy/sklearn test modules
# because numpy.testing is required internally by numpy itself.
# Only exclude top-level standalone test directories and metadata.
EXCLUDE_DIRS = {
    "__pycache__",
    "benchmarks",
    "docs", "doc",
    "examples",
}
# Folder names to skip ONLY at the top level of each package
TOP_LEVEL_EXCLUDE = {
    "tests", "test",          # pandas/sklearn top-level tests (safe to remove)
}
EXCLUDE_EXTS = {".pyc", ".pyo", ".so.debug"}

def build_zip() -> bytes:
    """
    Install dependencies into a temp folder, then zip them together
    with the handler file so Lambda has everything it needs.
    """
    import shutil, subprocess, tempfile
    print_step(f"Building Lambda zip from {HANDLER_FILE} + dependencies")

    if not os.path.exists(HANDLER_FILE):
        sys.exit(f"ERROR: {HANDLER_FILE} not found in current directory.")

    tmp_dir = tempfile.mkdtemp(prefix="lambda_pkg_")
    try:
        # Install deps into tmp_dir
        print(f"    Installing: {', '.join(DEPS)}")
        print(f"    Target dir: {tmp_dir}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--quiet",
            "--target", tmp_dir,
            "--only-binary=:all:",        # wheels only, no source builds
            "--platform", "manylinux2014_x86_64",  # Linux platform (Lambda OS)
            "--python-version", "3.11",            # match Lambda runtime
            "--implementation", "cp",              # CPython
            "--abi", "cp311",
            "--ignore-requires-python",
            *DEPS,
        ])

        buf = io.BytesIO()
        total_files = 0
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Handler file at root
            zf.write(HANDLER_FILE, arcname=HANDLER_FILE)
            # 2. All installed packages — carefully pruned
            for root, dirs, files in os.walk(tmp_dir):
                rel_root = os.path.relpath(root, tmp_dir)
                depth    = len(rel_root.split(os.sep))

                # Always skip these dirs at any depth
                dirs[:] = [
                    d for d in dirs
                    if d not in EXCLUDE_DIRS
                    and not d.endswith(".dist-info")
                    and not d.endswith(".egg-info")
                ]

                # At depth=2 (inside a package), only skip top-level test dirs
                # for pandas/sklearn — but NOT for numpy (needs numpy/testing)
                if depth == 2:
                    package_name = rel_root.split(os.sep)[0]
                    if package_name not in ("numpy",):
                        dirs[:] = [d for d in dirs if d not in TOP_LEVEL_EXCLUDE]

                for file in files:
                    ext = os.path.splitext(file)[1]
                    if ext in EXCLUDE_EXTS:
                        continue
                    full_path = os.path.join(root, file)
                    arcname   = os.path.relpath(full_path, tmp_dir)
                    zf.write(full_path, arcname=arcname)
                    total_files += 1

        buf.seek(0)
        data = buf.read()
        with open(LAMBDA_ZIP, "wb") as f:
            f.write(data)
        size_mb = len(data) / 1_000_000
        print(f"    Packaged {total_files} files")
        print(f"    {LAMBDA_ZIP} ready ({size_mb:.1f} MB)")
        if size_mb > 90:
            print(f"    WARNING: zip is large — upload via S3 (handled automatically)")
        return data
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── 4. Create / update Lambda ─────────────────────────────────────────────────
def deploy_lambda(role_arn: str, zip_bytes: bytes):
    print_step(f"Deploying Lambda function: {FUNCTION_NAME}")
    lam = client("lambda")

    env_vars = {
        "S3_BUCKET":           BUCKET,
        "TRAIN_KEY":           "Matches_cleaned.csv",
        "TEST_KEY":            "Matches_test.csv",
        "RESULTS_KEY":         "results/knn_results.json",
        "KNN_K":               "8",
        "N_WORKERS":           "4",
        "TEST_SAMPLE_FRAC":    "0.1",
        "LOCALSTACK_ENDPOINT": "http://host.docker.internal:4566",
    }

    # ── Upload zip to S3 first (avoids 50 MB direct-upload limit) ───────────
    zip_s3_key = "lambda/lambda_knn.zip"
    print(f"    Uploading zip to s3://{BUCKET}/{zip_s3_key} …")
    s3 = client("s3")
    s3.put_object(Bucket=BUCKET, Key=zip_s3_key, Body=zip_bytes)
    print(f"    Upload complete.")

    try:
        lam.get_function(FunctionName=FUNCTION_NAME)
        # Function exists → update code from S3
        lam.update_function_code(
            FunctionName=FUNCTION_NAME,
            S3Bucket=BUCKET,
            S3Key=zip_s3_key,
        )
        lam.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Timeout=TIMEOUT,
            MemorySize=MEMORY_MB,
            Environment={"Variables": env_vars},
        )
        print(f"    Function updated.")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        lam.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime=RUNTIME,
            Role=role_arn,
            Handler=HANDLER,
            Code={"S3Bucket": BUCKET, "S3Key": zip_s3_key},
            Timeout=TIMEOUT,
            MemorySize=MEMORY_MB,
            Environment={"Variables": env_vars},
            Description="KNN MapReduce football match predictor",
        )
        print(f"    Function created from S3.")

    # Wait until Active — LocalStack on Windows can be slow (up to 2 min)
    print("    Waiting for function to become Active …")
    waited, max_wait, interval = 0, 120, 3
    while waited < max_wait:
        cfg = lam.get_function(FunctionName=FUNCTION_NAME)["Configuration"]
        current_state = cfg.get("State", "Unknown")
        reason        = cfg.get("StateReason", "")
        print(f"    [{waited:3d}s] State={current_state}  {reason}")
        if current_state == "Active":
            print("    ✓ Function is Active and ready.")
            break
        if current_state == "Failed":
            print(f"    ✗ Function entered Failed state: {reason}")
            sys.exit(1)
        time.sleep(interval)
        waited += interval
    else:
        print(f"    Function still Pending after {max_wait}s.")
        print("    This is a LocalStack quirk on Windows — try invoking anyway:")
        print("      python aws_utils.py --wait-and-invoke --k 8 --sample 0.2")


# ── 5. Invoke Lambda ──────────────────────────────────────────────────────────
def invoke_lambda(payload: dict = None):
    print_step(f"Invoking Lambda: {FUNCTION_NAME}")
    lam     = client("lambda")
    payload = payload or {}
    resp    = lam.invoke(
        FunctionName=FUNCTION_NAME,
        InvocationType="RequestResponse",   # synchronous
        Payload=json.dumps(payload).encode(),
    )
    raw    = resp["Payload"].read()
    result = json.loads(raw)

    if "FunctionError" in resp:
        print(f"  Lambda ERROR: {resp['FunctionError']}")
        print(raw.decode())
        return

    body = json.loads(result.get("body", "{}"))
    print(f"\n  ✅ Accuracy : {body.get('accuracy')}")
    print(f"  ⏱  Elapsed  : {body.get('elapsed_seconds')}s")
    print(f"  🗂  Train N  : {body.get('n_train')}")
    print(f"  🧪 Test N   : {body.get('n_test')}")
    print(f"  K           : {body.get('k')}")
    print("\n  Classification report:")
    for cls, metrics in body.get("classification_report", {}).items():
        if isinstance(metrics, dict):
            print(f"    {cls:12s}  "
                  f"prec={metrics['precision']:.3f}  "
                  f"rec={metrics['recall']:.3f}  "
                  f"f1={metrics['f1-score']:.3f}")


# ── Fetch results from S3 ─────────────────────────────────────────────────────
def fetch_results():
    print_step("Fetching results from S3")
    s3  = client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key="results/knn_results.json")
    data = json.loads(obj["Body"].read())
    print(json.dumps(data, indent=2))
    return data


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LocalStack KNN setup & deploy")
    parser.add_argument("--invoke",   action="store_true", help="Invoke Lambda after deploy")
    parser.add_argument("--zip-only", action="store_true", help="Only build the zip, don't deploy")
    parser.add_argument("--fetch",    action="store_true", help="Fetch & print results from S3")
    args = parser.parse_args()

    if args.fetch:
        fetch_results()
        return

    if args.zip_only:
        build_zip()
        return

    ensure_bucket()
    role_arn  = setup_iam()
    zip_bytes = build_zip()
    deploy_lambda(role_arn, zip_bytes)

    if args.invoke:
        invoke_lambda()


if __name__ == "__main__":
    main()