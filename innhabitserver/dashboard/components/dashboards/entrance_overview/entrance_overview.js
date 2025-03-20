function setupEntranceCharts(data){
    const entranceChart = document.getElementById("entrance-chart").getContext("2d");

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

    new Chart(entranceChart, {
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

document.addEventListener("DOMContentLoaded", function() {
    const jsonElement = document.getElementById("entrance-json-element");
    const data = JSON.parse(jsonElement.dataset.json);
    setupEntranceCharts(data);
});
