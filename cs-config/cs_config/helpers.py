"""
Functions used to help OG-USA configure to COMP
"""
try:
    import boto3
except ImportError:
    boto3 = None
import gzip
import pandas as pd


def retrieve_puf(aws_access_key_id, aws_secret_access_key):
    """
    Function for retrieving the PUF from the OSPC S3 bucket
    """
    has_credentials = aws_access_key_id and aws_secret_access_key
    if has_credentials and boto3 is not None:
        client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        obj = client.get_object(Bucket="ospc-data-files", Key="puf.csv.gz")
        gz = gzip.GzipFile(fileobj=obj["Body"])
        puf_df = pd.read_csv(gz)
        return puf_df
    else:
        return None
