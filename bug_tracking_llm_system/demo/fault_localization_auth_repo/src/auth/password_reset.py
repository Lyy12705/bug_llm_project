from src.auth.email import send_email


def build_reset_message(user):
    return f"Reset requested for {user.email}"


def request_password_reset(user):
    message = build_reset_message(user)
    return {"email": user.email, "message": message}
