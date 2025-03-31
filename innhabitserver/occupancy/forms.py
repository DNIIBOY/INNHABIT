from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.forms import ChoiceField, DateField, Form, ModelChoiceField
from django.utils import timezone
from occupancy.models import Entrance

User = get_user_model()

event_types = [
    ("all", "All"),
    ("entries", "Entries"),
    ("exits", "Exits"),
]


class FilterEventsForm(Form):
    event_type = ChoiceField(choices=event_types, required=False)
    from_date = DateField(required=False)
    to_date = DateField(required=False)
    entrances = ModelChoiceField(queryset=Entrance.objects.all(), required=False)

    def clean_from_date(self) -> timezone.datetime | None:
        from_date = self.cleaned_data["from_date"]
        if from_date is None:
            return None
        if from_date > timezone.localtime().date():
            raise ValidationError("Cannot be in the future")
        return from_date

    def clean_to_date(self) -> timezone.datetime | None:
        to_date = self.cleaned_data["to_date"]
        if to_date is None:
            return None
        if to_date > timezone.localtime().date():
            raise ValidationError("Cannot be in the future")

        from_date = self.cleaned_data["from_date"]
        if from_date and to_date < from_date:
            raise ValidationError("Cannot be before from_date")
        return to_date
