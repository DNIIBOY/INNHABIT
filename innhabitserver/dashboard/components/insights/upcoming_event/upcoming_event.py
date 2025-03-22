from dashboard.models import LabelledDate
from django.http import Http404
from django.utils import timezone
from django_components import Component, register


@register("upcoming_event")
class UpcomingEvent(Component):
    template_name = "upcoming_event.html"

    def get_context_data(self) -> dict:
        today = timezone.localtime().date()
        labelled_date = (
            LabelledDate.objects.filter(date__gt=today).order_by("date").first()
        )
        if labelled_date is None:
            raise Http404("No upcoming event found")

        return {
            "date": labelled_date.date,
            "label": labelled_date.label,
            "diff": (labelled_date.date - today).days,
        }
