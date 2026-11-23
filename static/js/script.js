
// Mobile sidebar toggle
function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");

    if (!sidebar) return;

    if (window.innerWidth <= 768) {
        if (sidebar.style.display === "flex") {
            sidebar.style.display = "none";
        } else {
            sidebar.style.display = "flex";
        }
    }
}

// Close sidebar when a navigation link is selected
document.addEventListener("DOMContentLoaded", function () {
    const sidebar = document.getElementById("sidebar");
    const links = document.querySelectorAll(".nav-link");

    links.forEach(function (link) {
        link.addEventListener("click", function () {
            if (window.innerWidth <= 768 && sidebar) {
                sidebar.style.display = "none";
            }
        });
    });
});

// Automatically adjust sidebar on screen resize
window.addEventListener("resize", function () {
    const sidebar = document.getElementById("sidebar");

    if (!sidebar) return;

    if (window.innerWidth > 768) {
        sidebar.style.display = "flex";
    } else {
        sidebar.style.display = "none";
    }
});
