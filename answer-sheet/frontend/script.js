const uploadBtn = document.getElementById('uploadPdfBtn');
const useCameraBtn = document.getElementById('useCameraBtn');
const uploadSection = document.getElementById('uploadSection');
const cameraSection = document.getElementById('cameraSection');
const pdfInput = document.getElementById('pdfInput');
const video = document.getElementById('video');
const captureBtn = document.getElementById('captureBtn');
const canvas = document.getElementById('canvas');
const results = document.getElementById('results');
const registerNumber = document.getElementById('registerNumber');
const subjectCode = document.getElementById('subjectCode');

// Handle Upload PDF
uploadBtn.addEventListener('click', () => {
    uploadSection.classList.remove('hidden');
    cameraSection.classList.add('hidden');
    results.classList.add('hidden');
});

// Handle Use Camera
useCameraBtn.addEventListener('click', async () => {
    uploadSection.classList.add('hidden');
    cameraSection.classList.remove('hidden');
    results.classList.add('hidden');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        alert('Cannot access camera!');
    }
});

// Upload PDF or Image
pdfInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        sendImageToBackend(file);
    }
});

// Capture from Camera
captureBtn.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Stop camera
    const stream = video.srcObject;
    const tracks = stream.getTracks();
    tracks.forEach(track => track.stop());
    video.srcObject = null;

    canvas.toBlob(blob => {
        sendImageToBackend(blob);
    }, 'image/png');

    captureBtn.classList.add('hidden');
});

// Send image to backend
async function sendImageToBackend(imageBlob) {
    const formData = new FormData();
    formData.append('image', imageBlob);

    try {
        const response = await fetch('/extract_registration_number', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        displayResults(data.registration_number, data.subject_code);
    } catch (error) {
        alert('Error processing image.');
    }
}

// Display results
function displayResults(regNum, subCode) {
    registerNumber.textContent = `Register Number: ${regNum}`;
    subjectCode.textContent = `Subject Code: ${subCode}`;
    results.classList.remove('hidden');
}