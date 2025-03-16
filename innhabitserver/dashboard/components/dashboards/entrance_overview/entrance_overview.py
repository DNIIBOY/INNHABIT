from django_components import Component, register


@register("entrance_overview")
class EntranceOverview(Component):
    template_name = "entrance_overview.html"
    js_file = "entrance_overview.js"

    class Media:
        js = ["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.0.2/chart.min.js"]
