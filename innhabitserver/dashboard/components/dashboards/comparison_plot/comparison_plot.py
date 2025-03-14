from django_components import Component, register


@register("comparison_plot")
class ComparisonPlot(Component):
    template_name = "comparison_plot.html"
    js_file = "comparison_plot.js"

    class Media:
        js = ["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.0.2/chart.min.js"]
