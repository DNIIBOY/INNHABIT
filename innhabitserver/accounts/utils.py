import requests
from django.conf import settings
from django.contrib.auth.base_user import AbstractBaseUser
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
