const themeToggleBtn = document.getElementById('theme-toggle');
const body = document.body;

// Check for saved theme preference in localStorage on page load
const currentTheme = localStorage.getItem('theme');
if (currentTheme) {
    body.setAttribute('data-theme', currentTheme);
}

// Add event listener to the toggle button
themeToggleBtn.addEventListener('click', () => {
    // Get current theme
    const currentAttr = body.getAttribute('data-theme');

    // toggle between light and dark
    const newTheme = currentAttr === 'dark' ? 'light' : 'dark';
    body.setAttribute('data-theme', newTheme);

    // Save the user's preference in localStorage
    localStorage.setItem('theme', newTheme);
});
