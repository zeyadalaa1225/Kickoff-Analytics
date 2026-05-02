"""
aws_utils.py
────────────
Utility helpers for inspecting your LocalStack environment:
  - List S3 bucket contents
  - Describe Lambda functions
  - Tail / pretty-print KNN results
  - Trigger Lambda with custom parameters

Usage:
    python aws_utils.py --list-s3
    python aws_utils.py --list-lambdas
    python aws_utils.py --results
    python aws_utils.py --invoke --k 10 --sample 0.2
"""

import argparse
import json

import boto3
from botocore.exceptions import ClientError

ENDPOINT = "http://localhost:4566"
REGION   = "us-east-1"
BUCKET   = "kickoff-analytics"
FUNCTION = "knn-predictor"


def client(svc):
    return boto3.client(
        svc,
        endpoint_url=ENDPOINT,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name=REGION,
    )


# ── S3 ────────────────────────────────────────────────────────────────────────
def list_s3(bucket=BUCKET):
    s3 = client("s3")
    print(f"\nContents of s3://{bucket}/")
    print(f"{'Key':<50} {'Size':>12}  {'LastModified'}")
    print("─" * 80)
    try:
        resp = s3.list_objects_v2(Bucket=bucket)
        for obj in resp.get("Contents", []):
            size = obj["Size"]
            size_str = f"{size:,}" if size < 1_000_000 else f"{size/1_000_000:.1f} MB"
            print(f"{obj['Key']:<50} {size_str:>12}  {obj['LastModified'].strftime('%Y-%m-%d %H:%M')}")
    except ClientError as e:
        print(f"  Error: {e}")


def read_s3_json(key, bucket=BUCKET):
    s3  = client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


# ── Lambda ────────────────────────────────────────────────────────────────────
def list_lambdas():
    lam = client("lambda")
    print("\nLambda Functions")
    print("─" * 60)
    resp = lam.list_functions()
    for fn in resp["Functions"]:
        state  = fn.get("State", "–")
        mem    = fn.get("MemorySize", "–")
        to     = fn.get("Timeout", "–")
        print(f"  {fn['FunctionName']:<30}  State={state}  Mem={mem}MB  Timeout={to}s")
        print(f"  {'':30}  Runtime={fn['Runtime']}  Handler={fn['Handler']}")
        env = fn.get("Environment", {}).get("Variables", {})
        if env:
            for k, v in env.items():
                print(f"  {'':30}    {k}={v}")
        print()


def wait_for_active(lam, max_wait=120, interval=3):
    """Poll until function is Active or timeout."""
    waited = 0
    while waited < max_wait:
        cfg = lam.get_function(FunctionName=FUNCTION)["Configuration"]
        state = cfg.get("State", "Unknown")
        print(f"  [{waited:3d}s] Function state: {state}")
        if state == "Active":
            print("  ✓ Ready.")
            return True
        if state == "Failed":
            print(f"  ✗ Failed: {cfg.get('StateReason','')}")
            return False
        import time; time.sleep(interval)
        waited += interval
    print(f"  Still not Active after {max_wait}s — attempting invoke anyway.")
    return False


def invoke_lambda(k=8, sample=0.1, n_workers=4, wait=False):
    lam     = client("lambda")
    payload = {"k": k, "test_sample_frac": sample, "n_workers": n_workers}
    print(f"\nInvoking {FUNCTION} with payload: {payload}")
    if wait:
        print("  Waiting for function to become Active first …")
        wait_for_active(lam)
    resp = lam.invoke(
        FunctionName=FUNCTION,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode(),
    )
    raw = resp["Payload"].read()
    if "FunctionError" in resp:
        print(f"Lambda error: {resp['FunctionError']}")
        print(raw.decode())
        return

    result = json.loads(raw)
    body   = json.loads(result.get("body", "{}"))
    print(f"\n  Accuracy  : {body.get('accuracy')}")
    print(f"  Elapsed   : {body.get('elapsed_seconds')}s")
    print(f"  K         : {body.get('k')}  |  Train: {body.get('n_train')}  |  Test: {body.get('n_test')}")
    print("\n  Full report:")
    for cls, m in body.get("classification_report", {}).items():
        if isinstance(m, dict):
            print(f"    {cls:12s}  prec={m['precision']:.3f}  rec={m['recall']:.3f}  f1={m['f1-score']:.3f}  support={int(m['support'])}")


def print_results():
    try:
        data = read_s3_json("results/knn_results.json")
        print("\nKNN Results from S3")
        print("─" * 60)
        print(f"  Accuracy : {data['accuracy']}")
        print(f"  K        : {data['k']}")
        print(f"  Train N  : {data['n_train']}")
        print(f"  Test N   : {data['n_test']}")
        print(f"  Elapsed  : {data['elapsed_seconds']}s")
        print("\n  Classification Report:")
        for cls, m in data.get("classification_report", {}).items():
            if isinstance(m, dict):
                print(f"    {cls:12s}  prec={m['precision']:.3f}  "
                      f"rec={m['recall']:.3f}  f1={m['f1-score']:.3f}  "
                      f"support={int(m['support'])}")
    except ClientError:
        print("  No results found yet. Run --invoke first.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LocalStack KNN utilities")
    parser.add_argument("--list-s3",          action="store_true")
    parser.add_argument("--list-lambdas",     action="store_true")
    parser.add_argument("--results",          action="store_true")
    parser.add_argument("--invoke",           action="store_true", help="Invoke immediately")
    parser.add_argument("--wait-and-invoke",  action="store_true", help="Wait for Active then invoke")
    parser.add_argument("--k",                type=int,   default=8)
    parser.add_argument("--sample",           type=float, default=0.1)
    parser.add_argument("--workers",          type=int,   default=4)
    args = parser.parse_args()

    if args.list_s3:
        list_s3()
    if args.list_lambdas:
        list_lambdas()
    if args.results:
        print_results()
    if args.invoke:
        invoke_lambda(args.k, args.sample, args.workers, wait=False)
    if args.wait_and_invoke:
        invoke_lambda(args.k, args.sample, args.workers, wait=True)
    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()