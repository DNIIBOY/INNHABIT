var allTimeChart;

document.body.addEventListener("htmx:afterOnLoad", function(event) {
    if (event.detail.target.id !== "all-time-json-element") {
        return;
    }
    const jsonElement = document.getElementById("all-time-json-element");
    const data = JSON.parse(jsonElement.dataset.json);
    allTimeChart.data.labels = data["labels"];
    allTimeChart.data.datasets[0].data = data["occupancy_counts"];
    allTimeChart.update()
});

document.addEventListener("DOMContentLoaded", function() {
    const jsonElement = document.getElementById("all-time-json-element");
    const data = JSON.parse(jsonElement.dataset.json);
    setupAllTimePlot(data);
});

function setupAllTimePlot(data) {
    const chartElement = document.getElementById("all-time-chart").getContext("2d");

    const gradientStroke1 = chartElement.createLinearGradient(0, 230, 0, 50);
    gradientStroke1.addColorStop(1, "rgba(49, 169, 193, 0.2)");
    gradientStroke1.addColorStop(0.2, "rgba(72,72,176,0.0)");

    allTimeChart = new Chart(chartElement, {
        type: "line",
        data: {
            labels: data["labels"],
            datasets: [
                {
                    tension: 0.4,
                    borderWidth: 0,
                    pointRadius: 0,
                    borderColor: "#31A9C1",
                    borderWidth: 3,
                    backgroundColor: gradientStroke1,
                    fill: true,
                    data: data["occupancy_counts"],
                    maxBarThickness: 6,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
            },
            interaction: {
                intersect: false,
                mode: "index",
            },
            scales: {
                y: {
                    grid: {
                        drawBorder: false,
                        display: true,
                        drawOnChartArea: true,
                        drawTicks: false,
                        borderDash: [5, 5],
                    },
                    ticks: {
                        display: true,
                        padding: 10,
                        color: "#b2b9bf",
                        font: {
                            size: 11,
                            family: "Open Sans",
                            style: "normal",
                            lineHeight: 2,
                        },
                    },
                },
                x: {
                    grid: {
                        drawBorder: false,
                        display: false,
                        drawOnChartArea: false,
                        drawTicks: false,
                        borderDash: [5, 5],
                    },
                    ticks: {
                        display: true,
                        color: "#b2b9bf",
                        padding: 20,
                        font: {
                            size: 11,
                            family: "Open Sans",
                            style: "normal",
                            lineHeight: 2,
                        },
                    },
                },
            },
        },
    });
}
