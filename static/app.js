const input = document.getElementById("image-input");
const dropZone = document.getElementById("drop-zone");
const preview = document.getElementById("selected-preview");
const dropTitle = document.getElementById("drop-title");
const dropDetail = document.getElementById("drop-detail");
const form = document.getElementById("upload-form");
const submitButton = document.getElementById("submit-button");
let previewUrl = null;

function showSelection() {
  const file = input.files[0];
  if (!file) return;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  dropZone.classList.add("has-image");
  dropTitle.textContent = file.name;
  dropDetail.textContent = "Ready to analyze";
  submitButton.disabled = false;
  submitButton.firstElementChild.textContent = "Analyze image";
}

input.addEventListener("change", showSelection);
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");
  if (event.dataTransfer.files.length) {
    input.files = event.dataTransfer.files;
    showSelection();
  }
});
form.addEventListener("submit", () => {
  if (input.files.length) {
    submitButton.disabled = true;
    submitButton.classList.add("is-loading");
    submitButton.firstElementChild.textContent = "Analyzing image…";
  }
});
