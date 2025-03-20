from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent


@register("monthly_visitors")
class MonthlyVisitors(Component):
    template_name = "monthly_visitors.html"

    def get_context_data(self) -> dict:
        months = [
            "Januar",
            "Februar",
            "Marts",
            "April",
            "Maj",
            "Juni",
            "Juli",
            "August",
            "September",
            "Oktober",
            "November",
            "December",
        ]
        today = timezone.localtime().date()
        entries = EntryEvent.objects.filter(
            timestamp__date__year=today.year,
            timestamp__date__month=today.month,
        ).count()
        last_month_entries = EntryEvent.objects.filter(
            timestamp__date__year=today.year,
            timestamp__date__month=today.month - 1,
            timestamp__date__day__lte=today.day,
        ).count()

        if last_month_entries == 0:
            last_month_entries = 1  # Avoid division by zero

        return {
            "monthly_visitors": entries,
            "last_month_visitors": last_month_entries,
            "last_month_diff": entries - last_month_entries,
            "last_month_percentage": round(
                (entries - last_month_entries) / last_month_entries * 100, 2
            ),
            "date": today,
            "month": months[today.month - 1],
        }
