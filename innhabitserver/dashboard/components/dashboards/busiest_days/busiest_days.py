from dashboard.models import LabelledDate
from django.db.models import Count
from django.db.models.functions import TruncDate
from django.http import HttpRequest, HttpResponse
from django_components import Component, register
from occupancy.models import EntryEvent


@register("busiest_days")
class BusiestDays(Component):
    template_name = "busiest_days.html"

    def get(self, request: HttpRequest) -> HttpResponse:
        items = int(request.GET.get("items", 4))
        return self.render_to_response(request=request, kwargs={"items": items})

    def get_context_data(self, items: int = 5) -> dict:
        days_with_most_entries = (
            EntryEvent.objects.annotate(date=TruncDate("timestamp"))
            .values("date")
            .annotate(entry_count=Count("id"))
            .order_by("-entry_count")
            .values("date", "entry_count")[:items]
        )
        labelled_days = LabelledDate.objects.filter(
            date__in=(day["date"] for day in days_with_most_entries)
        ).all()
        day_labels = {day.date: day.label for day in labelled_days}

        days = [
            {
                "date": day["date"],
                "entry_count": day["entry_count"],
                "label": day_labels.get(day["date"]),
            }
            for day in days_with_most_entries
        ]
        return {
            "days": days,
        }
