from django.contrib.auth import get_user_model
from django.contrib.auth.forms import BaseUserCreationForm
from django.core.exceptions import ValidationError
from django.forms import CharField, EmailField, Form

User = get_user_model()


class AddUserForm(Form):
    email = EmailField(required=True)

    def clean_email(self) -> str:
        email = self.cleaned_data["email"]
        if User.objects.filter(email=email).exists():
            raise ValidationError("User with this email already exists.")

        return email


class SetupUserForm(BaseUserCreationForm):
    first_name = CharField(required=True)
    last_name = CharField(required=True)

    class Meta:
        model = User
        fields = (
            "first_name",
            "last_name",
        )
