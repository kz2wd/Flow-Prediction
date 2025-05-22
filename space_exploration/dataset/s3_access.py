import s3fs


# Set it in ~/.aws/credentials:
"""
[default]
aws_access_key_id = your-access-key
aws_secret_access_key = your-secret-key
"""

fs = s3fs.S3FileSystem(profile='default', client_kwargs={
    'endpoint_url': 'http://localhost:9000'  # Minio url here
})

def get_s3_map(file_path):
    return s3fs.S3Map(root=file_path, s3=fs, check=False)