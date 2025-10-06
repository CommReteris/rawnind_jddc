#!/usr/bin/env python3
"""
Smoke test script for RawNIND pipeline.

Downloads 3 scenes from the dataset, organizes them, runs prep_image_dataset,
interrupts it after a short time, and checks for partial results.
"""

import os
import subprocess
import signal
import time
import yaml
import shutil
import sys

def run_command(cmd, cwd=None, shell=False):
    """Run a shell command and return the process."""
    print(f"Running: {cmd}")
    try:
        return subprocess.Popen(cmd, shell=shell, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"Command failed to execute: {e}")
        sys.exit(1)

def main():
    # Step 1: Download 3 files to tmp
    print("Step 1: Downloading 3 files...")
    os.makedirs("tmp", exist_ok=True)
    os.chdir("tmp")

    # Use subprocess.run for better error handling and safety
    try:
        # Get file URLs using curl and jq
        curl_cmd = [
            "curl", "-s",
            "https://dataverse.uclouvain.be/api/datasets/:persistentId/?persistentId=doi:10.14428/DVN/DEQCIM"
        ]
        curl_result = subprocess.run(curl_cmd, capture_output=True, text=True)
        if curl_result.returncode != 0:
            print("Failed to get file list")
            sys.exit(1)

        # Parse with jq and extract wget commands
        jq_cmd = ["jq", "-r",
                 ".data.latestVersion.files[] | \"wget -c -O \\\"\\(.dataFile.filename)\\\" https://dataverse.uclouvain.be/api/access/datafile/\\(.dataFile.id)\""]
        jq_result = subprocess.run(jq_cmd, input=curl_result.stdout, capture_output=True, text=True)
        if jq_result.returncode != 0:
            print("Failed to parse file list")
            sys.exit(1)

        # Execute wget commands
        wget_commands = jq_result.stdout.split('\n')[:3]
        for wget_cmd in wget_commands:
            if wget_cmd.strip():
                subprocess.run(wget_cmd, shell=True, check=True)
    finally:
        os.chdir("..")

    # Check if files were downloaded
    flat_dir = "tmp"
    files = os.listdir(flat_dir) if os.path.exists(flat_dir) else []
    print(f"Downloaded files: {files}")
    if not files:
        print("No files downloaded")
        sys.exit(1)

    # Step 2: Organize files
    print("Step 2: Organizing files...")
    organize_cmd = ["bash", "./datasets/RawNIND/organize_files.sh", "tmp", "datasets/smoke_test"]
    proc = run_command(organize_cmd, shell=False)
    proc.wait()
    if proc.returncode != 0:
        print("Organize failed")
        sys.exit(1)

    # Step 3: Run prep_image_dataset and interrupt
    print("Step 3: Running prep_image_dataset and interrupting...")
    prep_cmd = [sys.executable, "src/rawnind/tools/prep_image_dataset.py", "--dataset", "smoke_test"]
    proc = subprocess.Popen(prep_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)  # Let it run for 5 seconds
    proc.send_signal(signal.SIGINT)  # Send keyboard interrupt
    proc.wait()

    # Step 4: Check for partial results
    print("Step 4: Checking for partial results...")
    yaml_path = "datasets/smoke_test/smoke_test_masks_and_alignments.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            try:
                data = yaml.safe_load(f)
                if data and len(data) > 0:
                    print(f"Success: Found {len(data)} partial results in {yaml_path}")
                else:
                    print("Failure: YAML exists but empty")
            except yaml.YAMLError:
                print("Failure: YAML file corrupted")
    else:
        print("Failure: No YAML file found")

    # Step 5: Cleanup
    print("Step 5: Cleaning up...")
    dirs_to_remove = ["tmp", "datasets/smoke_test"]
    for d in dirs_to_remove:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Removed {d}")

    print("Smoke test complete.")

if __name__ == "__main__":
    main()