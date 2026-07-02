from src.auth.validator import validate_token


def start_session(user):
    token = validate_token(user.token)
    return {"user_id": user.id, "token": token}
