function setupComparisonPlot(data) {
    const comparisonChart = document.getElementById("comparison-chart").getContext("2d");


    const gradientStroke1 = comparisonChart.createLinearGradient(0, 230, 0, 50);
    gradientStroke1.addColorStop(1, "rgba(49, 169, 193, 0.2)");
    gradientStroke1.addColorStop(0.2, "rgba(72,72,176,0.0)");

    const gradientStroke2 = comparisonChart.createLinearGradient(0, 230, 0, 50);
    gradientStroke2.addColorStop(1, "rgba(33, 26, 81, 0.2)");
    gradientStroke2.addColorStop(0.2, "rgba(72,72,176,0.0)");


    new Chart(comparisonChart, {
        type: "line",
        data: {
            labels: data["labels"],
            datasets: [
                {
                    label: "Idag",
                    tension: 0.4,
                    borderWidth: 0,
                    pointRadius: 0,
                    borderColor: "#31A9C1",
                    borderWidth: 3,
                    backgroundColor: gradientStroke1,
                    fill: true,
                    data: data["today_counts"],
                    maxBarThickness: 6,
                },
                {
                    label: "Gennemsnitlig",
                    tension: 0.4,
                    borderWidth: 0,
                    pointRadius: 0,
                    borderColor: "#201950",
                    borderWidth: 3,
                    backgroundColor: gradientStroke2,
                    fill: true,
                    data: [],
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

document.addEventListener("DOMContentLoaded", function() {
    const jsonElement = document.getElementById("comparison-json-element");
    const data = JSON.parse(jsonElement.dataset.json);
    setupComparisonPlot(data);
});
