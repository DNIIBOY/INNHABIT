from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.forms import EmailField, Form

User = get_user_model()


class AddUserForm(Form):
    email = EmailField(required=True)

    def clean_email(self) -> str:
        email = self.cleaned_data["email"]
        if User.objects.filter(email=email).exists():
            raise ValidationError("User with this email already exists.")

        return email
