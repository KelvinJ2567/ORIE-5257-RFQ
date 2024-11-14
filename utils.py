import hashlib

def hash_tuple(t: tuple) -> str:
    """Get the hash value of a tuple

    Args:
        t: The tuple to hash.

    Returns:
        str. The hash value of the tuple.
    """
    md5 = hashlib.md5()
    md5.update(str(t).encode())
    return md5.hexdigest()
