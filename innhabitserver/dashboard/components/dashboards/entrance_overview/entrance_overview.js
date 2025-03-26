var entranceChart;

document.body.addEventListener("htmx:afterOnLoad", function(event) {
    if (event.detail.target.id !== "entrance-json-element") {
        return;
    }
    const jsonElement = document.getElementById("entrance-json-element");
    const data = JSON.parse(jsonElement.dataset.json);
    entranceChart.data.labels = data["labels"];
    entranceChart.data.datasets[0].data = data["events"];
    entranceChart.data.datasets[0].backgroundColor = data["colors"];
    entranceChart.update()
});

document.addEventListener("DOMContentLoaded", function() {
    const jsonElement = document.getElementById("entrance-json-element");
    const data = JSON.parse(jsonElement.dataset.json);
    setupEntranceCharts(data);
});

function setupEntranceCharts(data){
    const chartElement = document.getElementById("entrance-chart").getContext("2d");

    const options = {
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
        }
    }

    entranceChart = new Chart(chartElement, {
        type: "doughnut",
        data: {
            labels: data["labels"],
            datasets: [
                {
                    data: data["events"],
                    backgroundColor: data["colors"],
                },
            ],
        },
        options: options,
    });
}
