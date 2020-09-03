"""
Functions used to help OG-USA configure to COMP
"""
try:
    import boto3
except ImportError:
    boto3 = None
import gzip
import pandas as pd
from taxcalc import Policy
from collections import defaultdict

TC_LAST_YEAR = Policy.LAST_BUDGET_YEAR

POLICY_SCHEMA = {
    "labels": {
        "year": {
            "type": "int",
            "validators": {
                "choice": {
                    "choices": [
                        yr for yr in range(2013, TC_LAST_YEAR + 1)
                    ]
                }
            }
        },
        "MARS": {
            "type": "str",
            "validators": {"choice": {"choices": ["single", "mjoint",
                                                  "mseparate", "headhh",
                                                  "widow"]}}
        },
        "idedtype": {
            "type": "str",
            "validators": {"choice": {"choices": ["med", "sltx", "retx", "cas",
                                                  "misc", "int", "char"]}}
        },
        "EIC": {
            "type": "str",
            "validators": {"choice": {"choices": ["0kids", "1kid",
                                                  "2kids", "3+kids"]}}
        },
        "data_source": {
            "type": "str",
            "validators": {"choice": {"choices": ["PUF", "CPS", "other"]}}
        }
    },
    "additional_members": {
        "section_1": {"type": "str"},
        "section_2": {"type": "str"},
        "start_year": {"type": "int"},
        "checkbox": {"type": "bool"}
    }
}


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
