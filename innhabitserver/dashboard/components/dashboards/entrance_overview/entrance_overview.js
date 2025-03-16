var ctx2 = document.getElementById("chart-circle").getContext("2d");
var ctx3 = document.getElementById("chart-circle2").getContext("2d");


new Chart(ctx2, {
    type: "doughnut",
    data: {
        labels: ["A", "B"],
        datasets: [
            {
                data: [25, 30],
                backgroundColor: ["#cb0c9f", "#3A416F"],
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
    },
});

new Chart(ctx3, {
    type: "doughnut",
    data: {
        labels: ["A", "B"],
        datasets: [
            {
                data: [45, 30],
                backgroundColor: ["#cb0c9f", "#3A416F"],
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
    },
});
