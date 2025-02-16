import runpod
import os

# Set your API key
runpod.api_key = "rpa_Z8WR31CW8P8DNLITPTE0HEA064021AYFVVIDYBKI4527b3"  # Replace with your API key

# Set up your RunPod endpoint
endpoint = runpod.Endpoint("o1ukxcxrgpbbzk")  # Replace with your actual endpoint ID

# Input payload (modify as needed)
input_payload = {
    "input": {
        "command": "python /app/prepare_lora.py"
    }
}

# Run the training job asynchronously
try:
    run_request = endpoint.run(input_payload)

    # Print job ID for tracking
    print(f"Job started: {run_request.job_id}")

    # Polling job status
    while True:
        status = run_request.status()
        print(f"Current job status: {status}")

        if status == "COMPLETED":
            output = run_request.output()
            print("Job output:", output)
            break  # Exit loop when job is done
        elif status in ["FAILED", "ERROR"]:
            print("Job failed.")
            break
        else:
            print("Job is still in progress... Checking again in 5 seconds.")
            import time
            time.sleep(5)  # Wait before checking again

except Exception as e:
    print(f"An error occurred: {e}")
