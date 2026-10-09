(() => {
  const $ = (id) => document.getElementById(id);
  const overview = $("overviewPage");
  const placeholder = $("placeholderPage");
  const title = $("pageTitle");
  const placeholderTitle = $("placeholderTitle");
  const placeholderText = $("placeholderText");
  const sidebar = $("sidebar");
  const backdrop = $("modalBackdrop");
  const toast = $("toast");
  let toastTimer;

  const pageCopy = {
    "Projects": "Browse, organize, and reopen your recap projects.",
    "Create Recap": "Prepare a new movie recap project.",
    "Render Queue": "Track video jobs and their processing status.",
    "Usage & Plan": "Review usage limits and your workspace plan.",
    "Settings": "Manage workspace preferences and account settings."
  };

  function showToast(message) {
    toast.textContent = message;
    toast.classList.remove("hidden");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.add("hidden"), 3200);
  }

  function openModal() {
    backdrop.classList.remove("hidden");
    $("projectName").focus();
  }
  function closeModal() {
    backdrop.classList.add("hidden");
  }
  function goToPage(page) {
    document.querySelectorAll(".nav-item").forEach((button) => {
      button.classList.toggle("active", button.dataset.page === page);
    });
    title.textContent = page;
    sidebar.classList.remove("open");
    if (page === "Overview") {
      overview.classList.remove("hidden");
      placeholder.classList.add("hidden");
      return;
    }
    if (page === "Create Recap") {
      openModal();
      return;
    }
    overview.classList.add("hidden");
    placeholder.classList.remove("hidden");
    placeholderTitle.textContent = page;
    placeholderText.textContent = pageCopy[page] || "This page will be connected in a later implementation step.";
  }

  document.querySelectorAll("[data-page]").forEach((button) => {
    button.addEventListener("click", () => goToPage(button.dataset.page));
  });
  $("newProjectButton").addEventListener("click", openModal);
  $("closeModal").addEventListener("click", closeModal);
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) closeModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeModal();
  });
  $("projectForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const projectName = $("projectName").value.trim();
    if (!projectName) return;
    closeModal();
    $("projectName").value = "";
    showToast('Draft "' + projectName + '" created in this preview. No render job was started.');
  });
  $("returnOverview").addEventListener("click", () => goToPage("Overview"));
  $("mobileMenu").addEventListener("click", () => sidebar.classList.toggle("open"));
  $("helpButton").addEventListener("click", () => showToast("Help center will be added after the core workspace is connected."));
  $("learnMoreButton").addEventListener("click", () => showToast("Invite-only access will be enforced by the authentication backend, not this prototype."));
  $("profileButton").addEventListener("click", () => showToast("Admin profile settings are not connected yet."));
  $("topProfile").addEventListener("click", () => showToast("Account login will be connected in the authentication step."));
  $("activityOptions").addEventListener("click", () => showToast("Activity feed is sample data for the frontend prototype."));
  document.querySelectorAll(".row-more").forEach((button) => {
    button.addEventListener("click", () => showToast("Project actions will be connected when project data is backed by the database."));
  });
})();