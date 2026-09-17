const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewText = document.getElementById("previewText");
const predictButton = document.getElementById("predictButton");
const resultText = document.getElementById("resultText");


// Show selected image
imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    const imageURL = URL.createObjectURL(file);

    imagePreview.src = imageURL;
    imagePreview.style.display = "block";
    previewText.style.display = "none";
});


// Send image to Flask for prediction
predictButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    // Check image selection
    if (!file) {
        resultText.textContent = "Please select a mango image first.";
        return;
    }

    // Show loading message
    resultText.textContent = "Analyzing image...";

    // Create form data
    const formData = new FormData();
    formData.append("image", file);

    try {

        // Send image to Flask
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        // Handle error
        if (!response.ok) {
            resultText.textContent = data.error || "Prediction failed.";
            return;
        }

        // Show prediction
        resultText.innerHTML = `
            <strong>Disease:</strong> ${data.prediction}<br>
            <strong>Confidence:</strong> ${data.confidence}%
        `;

    } catch (error) {

        resultText.textContent =
            "Something went wrong while connecting to the server.";

        console.error(error);
    }
});