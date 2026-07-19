from .validator import validate_token


def start_session(user):
    return validate_token(user.token)
