const entriesChart = document.getElementById("entries-chart").getContext("2d");
const exitsCharts = document.getElementById("exits-chart").getContext("2d");

const jsonElement = document.getElementById("json-element");
const data = JSON.parse(jsonElement.dataset.json);

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

new Chart(entriesChart, {
    type: "doughnut",
    data: {
        labels: data["labels"],
        datasets: [
            {
                data: data["entries"],
                backgroundColor: data["colors"],
            },
        ],
    },
    options: options,
});

new Chart(exitsCharts, {
    type: "doughnut",
    data: {
        labels: data["labels"],
        datasets: [
            {
                data: data["exits"],
                backgroundColor: data["colors"],
            },
        ],
    },
    options: options,
});
