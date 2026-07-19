DEFAULT_SESSION_TTL = 3600


def auth_feature_flags():
    return {"password_reset": True, "session_refresh": True}
