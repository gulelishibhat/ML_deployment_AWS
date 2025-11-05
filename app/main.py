import boto3
import pandas as pd
from inference import ModelWrapper
import io
from datetime import datetime

# Initialize model once — reused across Lambda invocations
model = ModelWrapper()

# Initialize S3 client
s3 = boto3.client("s3")

# Bucket name
BUCKET = "test-deployement-s3"
INPUT_PREFIX = "input/"     # Folder where feature CSVs are stored
OUTPUT_PREFIX = "output/"   # Folder where predictions will go

def lambda_handler(event, context):
    try:
        # List objects in input folder
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=INPUT_PREFIX)
        if "Contents" not in response:
            return {"status": "error", "message": "No input files found in S3."}

        # Select the latest file by LastModified
        latest_file = max(response["Contents"], key=lambda x: x["LastModified"])
        latest_key = latest_file["Key"]

        # Download latest CSV
        csv_obj = s3.get_object(Bucket=BUCKET, Key=latest_key)
        data = pd.read_csv(csv_obj["Body"])

        # Run predictions
        features = data.values.tolist()
        predictions = [int(model.predict(row)) for row in features]

        # Prepare output CSV
        output_df = pd.DataFrame({"prediction": predictions})

        # Construct output key with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        output_key = f"{OUTPUT_PREFIX}predictions_{timestamp}.csv"

        # Upload predictions back to S3
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=BUCKET, Key=output_key, Body=csv_buffer.getvalue())

        return {
            "status": "success",
            "input_file": latest_key,
            "output_file": output_key
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
