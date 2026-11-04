# heal-guard/world/tools.py
from typing import Dict, List, Any, Optional

def list_buckets(limit: int = 10, next_token: Optional[str] = None) -> Dict[str, Any]:
    """Lists S3 buckets. Paginated.
    
    Args:
        limit: Max number of buckets to return.
        next_token: Token for pagination.
    """
    pass

def get_bucket_policy(bucket: str) -> Dict[str, Any]:
    """Gets the bucket policy.
    
    Args:
        bucket: The name of the bucket.
    """
    pass

def get_bucket_acl(bucket: str) -> Dict[str, Any]:
    """Gets the bucket ACL.
    
    Args:
        bucket: The name of the bucket.
    """
    pass

def get_public_access_block(bucket: str) -> Dict[str, Any]:
    """Gets the public access block configuration.
    
    Args:
        bucket: The name of the bucket.
    """
    pass

def list_iam_roles(limit: int = 10) -> Dict[str, Any]:
    """Lists IAM roles.
    
    Args:
        limit: Max number of roles to return.
    """
    pass

def get_role_policies(role_name: str) -> Dict[str, Any]:
    """Gets inline policies for an IAM role.
    
    Args:
        role_name: The name of the role.
    """
    pass

def describe_security_groups(group_ids: List[str] = None) -> Dict[str, Any]:
    """Describes security groups.
    
    Args:
        group_ids: Optional list of group IDs to filter by.
    """
    pass

def finish(answer: List[str]) -> Dict[str, Any]:
    """Submits the final answer containing a list of vulnerable resource IDs.
    
    Args:
        answer: The list of vulnerable resource IDs (e.g., ["bucket-123", "sg-456"]).
    """
    pass