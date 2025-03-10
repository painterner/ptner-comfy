import boto3
# import magic // 从文件内容推动文件mimeType
import mimetypes
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Configuration
R2_ACCESS_KEY = "e308d609df97024986bef89135a197d7"
R2_SECRET_KEY = "3851731c2fe5c48572492395e05b956c986b49d0cec4813568898237d4eeee9b"
R2_ENDPOINT_URL = "https://1735aad300849b0eec57f505946f8a0e.r2.cloudflarestorage.com"  # Example: https://<account_id>.r2.cloudflarestorage.com
R2_BUCKET_NAME = "video"

def upload_to_r2(upload_bucket, file_path, object_name, mimeType):
    try:
        print(f"File '{file_path}' uploading to R2 bucket '{upload_bucket}' as '{object_name}'.")
        # Initialize S3 client with R2 credentials and endpoint
        # mime = magic.Magic(mime=True)
        # mime_type = mime.from_file(file_path)
        # mime_type, encoding = mimetypes.guess_type(file_path)
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            endpoint_url=R2_ENDPOINT_URL
        )

        # Upload file
        with open(file_path, 'rb') as file:
            s3_client.upload_fileobj(file, upload_bucket, object_name, ExtraArgs={'ContentType': mimeType})
        
        print(f"File '{file_path}' uploaded to R2 bucket.")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except NoCredentialsError:
        print("Error: Missing credentials for R2.")
    except PartialCredentialsError:
        print("Error: Incomplete credentials provided for R2.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")