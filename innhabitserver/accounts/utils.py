import json

import requests
from django.conf import settings
from django.contrib.admin.models import LogEntry
from django.contrib.auth.base_user import AbstractBaseUser
from django.contrib.contenttypes.models import ContentType
from django.db.models import Model
from django.urls import reverse
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode

from .tokens import account_activation_token


def send_activation_email(
    user: AbstractBaseUser, host: str = "https://innhabit.dk"
) -> None:
    b64uid = urlsafe_base64_encode(force_bytes(user.pk))
    token = account_activation_token.make_token(user)
    path = reverse("activate_account", args=[b64uid, token])

    x = requests.post(
        "https://api.eu.mailgun.net/v3/mg.sigmaboy.dk/messages",
        auth=("api", settings.MAILGUN_API_KEY),
        data={
            "from": "noreply@sigmaboy.dk",
            "to": [user.email],
            "subject": "Innhabit - Konto aktivering",
            "text": f"Tryk her for at aktivere din konto: {host}{path}",
        },
        timeout=5,
    )
    print(x.text)


def log_admin_action(
    user: AbstractBaseUser, obj: Model, action_flag: int, message: list | str = ""
) -> LogEntry:
    """
    Logs an action performed by an admin user.
    :param user: The admin user performing the action.
    :param obj: The object being modified.
    :param action_flag: ADDITION, CHANGE, or DELETION.
    :param message: Optional message describing the change.
    """
    if isinstance(message, list):
        message = json.dumps(message)
    return LogEntry.objects.create(
        user_id=user.id,
        content_type=ContentType.objects.get_for_model(obj.__class__),
        object_id=obj.pk,
        object_repr=str(obj),
        action_flag=action_flag,
        change_message=message,
    )
