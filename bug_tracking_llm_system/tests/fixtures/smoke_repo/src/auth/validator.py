def validate_token(token):
    """Return a normalized token.

    This intentionally reproduces the missing-token failure used by the local
    pipeline and fault-localization smoke checks.
    """

    return token.strip()
