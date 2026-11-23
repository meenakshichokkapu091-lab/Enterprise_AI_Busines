
function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");

    if (sidebar) {
        sidebar.classList.toggle("show");
    }
}

// Close mobile sidebar when clicking outside
document.addEventListener("click", function(event) {
    const sidebar = document.getElementById("sidebar");
    const menuButton = document.querySelector(".menu-toggle");

    if (
        sidebar &&
        menuButton &&
        window.innerWidth <= 750 &&
        !sidebar.contains(event.target) &&
        !menuButton.contains(event.target)
    ) {
        sidebar.classList.remove("show");
    }
});

// Table search
function searchTable(inputId, tableId) {
    const input = document.getElementById(inputId);
    const table = document.getElementById(tableId);

    if (!input || !table) return;

    const query = input.value.toLowerCase();

    table.querySelectorAll("tbody tr").forEach(row => {
        row.style.display = row.innerText.toLowerCase().includes(query)
            ? ""
            : "none";
    });
}

// Download table as CSV
function downloadTableCSV(tableId, filename) {
    const table = document.getElementById(tableId);

    if (!table) return;

    const rows = Array.from(table.querySelectorAll("tr"));

    const csv = rows.map(row =>
        Array.from(row.querySelectorAll("th, td"))
            .map(cell =>
                '"' + cell.innerText.replace(/"/g, '""') + '"'
            )
            .join(",")
    ).join("\n");

    const blob = new Blob([csv], {
        type: "text/csv;charset=utf-8;"
    });

    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename || "report.csv";
    link.click();

    URL.revokeObjectURL(link.href);
}

// Chart download
function downloadChart(chartId, filename) {
    const canvas = document.getElementById(chartId);

    if (!canvas) return;

    const link = document.createElement("a");
    link.download = filename || "chart.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
}

