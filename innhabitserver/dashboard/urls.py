from dashboard.components.dashboards import (
    AllTimePlot,
    BusiestDays,
    ComparisonPlot,
    DailyVisitors,
    EntranceOverview,
    LatestEvents,
    MonthlyVisitors,
    Occupants,
    WeeklyVisitors,
)
from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="dashboard"),
    path("top_row/", views.top_row, name="dashboard_top_row"),
    path("insights/", views.insights, name="insights"),
    path("insights_dashboard/", views.insights_dashboard, name="insights_dashboard"),
    path("components/all_time_plot/", AllTimePlot.as_view(), name="all_time_plot"),
    path("components/busiest_days/", BusiestDays.as_view(), name="busiest_days"),
    path(
        "components/comparison_plot/", ComparisonPlot.as_view(), name="comparison_plot"
    ),
    path("components/daily_visitors/", DailyVisitors.as_view(), name="daily_visitors"),
    path(
        "components/entrance_overview/",
        EntranceOverview.as_view(),
        name="entrance_overview",
    ),
    path("components/latest_events/", LatestEvents.as_view(), name="latest_events"),
    path(
        "components/monthly_visitors/",
        MonthlyVisitors.as_view(),
        name="monthly_visitors",
    ),
    path("components/occupants/", Occupants.as_view(), name="occupants"),
    path(
        "components/weekly_visitors/", WeeklyVisitors.as_view(), name="weekly_visitors"
    ),
]
