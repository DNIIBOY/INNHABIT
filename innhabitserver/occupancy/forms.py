from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.forms import (
    BooleanField,
    CharField,
    ChoiceField,
    DateField,
    Form,
    IntegerField,
    ModelMultipleChoiceField,
)
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
    entrances = ModelMultipleChoiceField(
        queryset=Entrance.objects.all(), required=False
    )
    test_events = BooleanField(required=False, initial=False)

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


class ConfigureDeviceForm(Form):
    entry_box_keys = [
        "entry_top_left_x",
        "entry_top_left_y",
        "entry_bottom_right_x",
        "entry_bottom_right_y",
    ]
    exit_box_keys = [
        "exit_top_left_x",
        "exit_top_left_y",
        "exit_bottom_right_x",
        "exit_bottom_right_y",
    ]

    request_image = BooleanField(required=False, initial=False)
    entry_top_left_x = IntegerField(required=False, min_value=0)
    entry_top_left_y = IntegerField(required=False, min_value=0)
    entry_bottom_right_x = IntegerField(required=False, min_value=0)
    entry_bottom_right_y = IntegerField(required=False, min_value=0)

    exit_top_left_x = IntegerField(required=False, min_value=0)
    exit_top_left_y = IntegerField(required=False, min_value=0)
    exit_bottom_right_x = IntegerField(required=False, min_value=0)
    exit_bottom_right_y = IntegerField(required=False, min_value=0)

    def clean(self) -> dict:
        cleaned_data = super().clean()
        has_entry_box = all(
            cleaned_data[key] is not None for key in self.entry_box_keys
        )
        if (
            any(cleaned_data[key] is not None for key in self.entry_box_keys)
            and not has_entry_box
        ):
            self.add_error(None, "All entry box fields are required")

        has_exit_box = all(cleaned_data[key] is not None for key in self.exit_box_keys)
        if (
            any(cleaned_data[key] is not None for key in self.exit_box_keys)
            and not has_exit_box
        ):
            self.add_error(None, "All exit box fields are required")

        if has_entry_box:
            if cleaned_data["entry_top_left_x"] >= cleaned_data["entry_bottom_right_x"]:
                self.add_error(
                    None, "Entry top left x must be less than bottom right x"
                )
            if cleaned_data["entry_top_left_y"] >= cleaned_data["entry_bottom_right_y"]:
                self.add_error(
                    None, "Entry top left y must be less than bottom right y"
                )

        if has_exit_box:
            if cleaned_data["exit_top_left_x"] >= cleaned_data["exit_bottom_right_x"]:
                self.add_error(None, "Exit top left x must be less than bottom right x")
            if cleaned_data["exit_top_left_y"] >= cleaned_data["exit_bottom_right_y"]:
                self.add_error(None, "Exit top left y must be less than bottom right y")

        return cleaned_data

    @property
    def entry_box(self) -> list[str] | None:
        box = [self.cleaned_data[key] for key in self.entry_box_keys]
        if not all(b is not None for b in box):
            return None
        return box

    @property
    def exit_box(self) -> list[str] | None:
        box = [self.cleaned_data[key] for key in self.exit_box_keys]
        if not all(b is not None for b in box):
            return None
        return box


class LabelledDateForm(Form):
    date = DateField()
    label = CharField(max_length=100)
