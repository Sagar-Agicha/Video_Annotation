const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const imageUpload = document.getElementById("imageUpload");
const videoUpload = document.getElementById("videoUpload");
const output = document.getElementById("output");
const saveButton = document.getElementById("save");
const prevButton = document.getElementById("prevImage");
const nextButton = document.getElementById("nextImage");
const imageCounter = document.getElementById("imageCounter");

let images = [];
let currentImageIndex = 0;
let currentImage = null;
let isDrawing = false;
let startX, startY;
let currentAnnotation = null;
let tempAnnotation = null;

const CANVAS_CONTAINER_WIDTH = 800;
const CANVAS_CONTAINER_HEIGHT = 600;

let selectedVideoFile = null;
let uploadedVideoFilename = null;

let canvasScale = {
    x: 1,
    y: 1
};

function initializeApp() {
    loadImages();
    
    const canvas = document.getElementById('canvas');
    const confirmClass = document.getElementById('confirmClass');
    const cancelAnnotation = document.getElementById('cancelAnnotation');
    
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', endDrawing);
    canvas.addEventListener('mouseout', endDrawing);
    
    confirmClass.addEventListener('click', saveAnnotation);
    cancelAnnotation.addEventListener('click', () => {
        currentAnnotation = null;
        document.getElementById('classPopup').style.display = 'none';
        drawAnnotations();
    });
    
    // Image upload handler
    document.getElementById('imageUpload').addEventListener('change', uploadImages);
    
    // Navigation buttons
    document.getElementById('prevImage').addEventListener('click', showPreviousImage);
    document.getElementById('nextImage').addEventListener('click', showNextImage);
    
    setupVideoUpload();
    
    // Replace save button with clear annotations
    document.getElementById('clearAnnotations').addEventListener('click', clearAnnotations);
    document.getElementById('startTraining').addEventListener('click', startTraining);
}

async function uploadImages(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    // First save current annotations if there's a current image
    if (currentImageIndex !== null && images[currentImageIndex]) {
        await saveAnnotationsToFile(currentImageIndex);
    }

    const formData = new FormData();
    for (let file of files) {
        formData.append('files[]', file);
    }

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                // Keep track of current annotations
                const existingAnnotations = [...images];
                
                // Reload all images
                await loadImages();
                
                // Restore annotations for existing images
                images = images.map(newImage => {
                    const existingImage = existingAnnotations.find(img => img.name === newImage.name);
                    if (existingImage && existingImage.annotations) {
                        return {
                            ...newImage,
                            annotations: existingImage.annotations
                        };
                    }
                    return newImage;
                });

                // Update display
                displayImage(currentImageIndex);
            }
        } else {
            console.error('Upload failed');
        }
    } catch (error) {
        console.error('Error uploading files:', error);
    }

    // Reset file input
    e.target.value = '';
}

async function loadImages() {
    try {
        const response = await fetch('/get_images');
        const data = await response.json();
        
        // Load annotations for each image
        const loadedImages = await Promise.all(data.images.map(async (img) => {
            try {
                const annotationResponse = await fetch(`/get_annotations/${img.name}`);
                const annotationData = await annotationResponse.json();
                return {
                    ...img,
                    annotations: annotationData.annotations || []
                };
            } catch (error) {
                console.error(`Error loading annotations for ${img.name}:`, error);
                return {
                    ...img,
                    annotations: []
                };
            }
        }));

        images = loadedImages;
        
        if (images.length > 0) {
            if (currentImageIndex === null || currentImageIndex >= images.length) {
                currentImageIndex = 0;
            }
            displayImage(currentImageIndex);
        }
        
        updateImageCounter();
    } catch (error) {
        console.error('Error loading images:', error);
    }
}

function updateImageCounter() {
    const counter = document.getElementById('imageCounter');
    counter.textContent = images.length > 0 
        ? `Image ${currentImageIndex + 1} of ${images.length}`
        : 'No images';
}

function setupEventListeners() {
    const canvas = document.getElementById('canvas');
    const classPopup = document.getElementById('classPopup');
    const confirmClass = document.getElementById('confirmClass');
    const cancelAnnotation = document.getElementById('cancelAnnotation');
    
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', endDrawing);
    canvas.addEventListener('mouseout', endDrawing);
    
    confirmClass.addEventListener('click', () => {
        const className = document.getElementById('className').value.trim();
        if (className) {
            saveAnnotation(className);
            classPopup.style.display = 'none';
        }
    });
    
    cancelAnnotation.addEventListener('click', () => {
        tempAnnotation = null;
        classPopup.style.display = 'none';
        redrawCanvas();
    });
}

function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    startX = (e.clientX - rect.left) * scaleX;
    startY = (e.clientY - rect.top) * scaleY;
    
    currentAnnotation = {
        x: startX,
        y: startY,
        width: 0,
        height: 0
    };
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const currentX = (e.clientX - rect.left) * scaleX;
    const currentY = (e.clientY - rect.top) * scaleY;
    
    currentAnnotation = {
        x: Math.min(startX, currentX),
        y: Math.min(startY, currentY),
        width: Math.abs(currentX - startX),
        height: Math.abs(currentY - startY)
    };
    
    drawAnnotations();
}

function endDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;
    
    if (currentAnnotation && 
        currentAnnotation.width > 5 && 
        currentAnnotation.height > 5) {
        showClassPopup();
    } else {
        currentAnnotation = null;
        drawAnnotations();
    }
}

function showClassPopup() {
    console.log('Opening popup with currentAnnotation:', currentAnnotation);
    const popup = document.getElementById('classPopup');
    const classInput = document.getElementById('className');
    
    if (!popup || !classInput) {
        console.error('Required popup elements not found');
        return;
    }
    
    popup.style.display = 'flex';
    classInput.value = '';
    classInput.focus();
}

function hideClassPopup() {
    const popup = document.getElementById('classPopup');
    const classInput = document.getElementById('className');
    
    if (popup) {
        popup.style.display = 'none';
    }
    if (classInput) {
        classInput.value = '';
    }
}

function saveAnnotation(className) {
    console.log('Attempting to save annotation:', { className, currentAnnotation });
    
    if (!currentAnnotation) {
        console.error('No current annotation to save');
        return;
    }
    
    if (typeof className !== 'string' || !className) {
        console.error('Invalid class name:', className);
        return;
    }

    const canvas = document.getElementById('canvas');
    
    // Create the normalized annotation
    const annotation = {
        x: currentAnnotation.x / canvas.width,
        y: currentAnnotation.y / canvas.height,
        width: currentAnnotation.width / canvas.width,
        height: currentAnnotation.height / canvas.height,
        class: className
    };

    console.log('Created normalized annotation:', annotation);

    // Initialize annotations array if needed
    if (!images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations = [];
    }

    // Add the annotation
    images[currentImageIndex].annotations.push(annotation);

    // Update display and clean up
    hideClassPopup();
    drawAnnotations();
    updateAnnotationsList();
    saveAnnotationsToFile();

    // Reset current annotation
    currentAnnotation = null;
}

function updateAnnotationsList() {
    const tbody = document.getElementById('annotationsList');
    tbody.innerHTML = '';
    
    const annotations = images[currentImageIndex].annotations || [];
    annotations.forEach((ann, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${ann.class}</td>
            <td>x: ${ann.x.toFixed(3)}, y: ${ann.y.toFixed(3)}, w: ${ann.width.toFixed(3)}, h: ${ann.height.toFixed(3)}</td>
            <td>
                <button class="action-btn edit-btn" onclick="editAnnotation(${index})">Edit</button>
                <button class="action-btn delete-btn" onclick="deleteAnnotation(${index})">Delete</button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function deleteAnnotation(index) {
    images[currentImageIndex].annotations.splice(index, 1);
    updateAnnotationsList();
    drawAnnotations();
}

function editAnnotation(index) {
    const annotation = images[currentImageIndex].annotations[index];
    document.getElementById('className').value = annotation.class;
    document.getElementById('classPopup').style.display = 'block';
    
    // Store the index being edited
    tempAnnotation = annotation;
    images[currentImageIndex].annotations.splice(index, 1);
    redrawCanvas();
}

function redrawCanvas() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas and redraw image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (currentImage) {
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    }
    
    // Redraw all annotations
    if (images[currentImageIndex].annotations) {
        drawAnnotations(images[currentImageIndex].annotations);
    }
}

function drawBox(ctx, box, className = '') {
    const canvasWidth = ctx.canvas.width;
    const canvasHeight = ctx.canvas.height;
    
    let x, y, width, height;
    
    if (box.normalized) {
        // Convert normalized coordinates to canvas coordinates
        width = box.width * canvasWidth;
        height = box.height * canvasHeight;
        x = box.x * canvasWidth;
        y = box.y * canvasHeight;
    } else {
        x = box.x;
        y = box.y;
        width = box.width;
        height = box.height;
    }

    // Draw rectangle
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#2196F3';
    ctx.setLineDash([]);
    ctx.rect(x, y, width, height);
    ctx.stroke();

    // Draw class name if provided
    if (className) {
        ctx.font = '14px Arial';
        const textWidth = ctx.measureText(className).width;
        const padding = 4;
        
        // Draw background for text
        ctx.fillStyle = '#2196F3';
        ctx.fillRect(x, y - 20, textWidth + (padding * 2), 20);
        
        // Draw text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(className, x + padding, y - 5);
    }
}

function drawAnnotations() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    // Clear and redraw image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (currentImage) {
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    }

    // Draw existing annotations
    if (images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations.forEach(ann => {
            const x = ann.x * canvas.width;
            const y = ann.y * canvas.height;
            const width = ann.width * canvas.width;
            const height = ann.height * canvas.height;

            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, width, height);

            // Draw label
            ctx.fillStyle = '#00ff00';
            ctx.font = '14px Arial';
            ctx.fillText(ann.class, x, y - 5);
        });
    }

    // Draw current annotation if any
    if (currentAnnotation) {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            currentAnnotation.x,
            currentAnnotation.y,
            currentAnnotation.width,
            currentAnnotation.height
        );
    }
}

function displayImage(index) {
    if (images.length === 0 || index < 0 || index >= images.length) return;

    const img = new Image();
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const canvasContainer = document.getElementById('canvas-container');

    img.onload = function() {
        currentImage = img;
        
        // Calculate dimensions maintaining aspect ratio
        const containerWidth = canvasContainer.clientWidth;
        const containerHeight = canvasContainer.clientHeight;
        const imageAspectRatio = img.width / img.height;
        const containerAspectRatio = containerWidth / containerHeight;
        
        let renderWidth, renderHeight;
        
        if (imageAspectRatio > containerAspectRatio) {
            renderWidth = containerWidth;
            renderHeight = containerWidth / imageAspectRatio;
        } else {
            renderHeight = containerHeight;
            renderWidth = containerHeight * imageAspectRatio;
        }
        
        // Set canvas dimensions
        canvas.width = renderWidth;
        canvas.height = renderHeight;
        
        // Draw image
        ctx.drawImage(img, 0, 0, renderWidth, renderHeight);
        
        // Draw annotations
        drawAnnotations();
        updateAnnotationsList();
    };

    img.src = images[index].url;
    currentImageIndex = index;
    updateImageCounter();
}

function showNextImage() {
    if (currentImageIndex < images.length - 1) {
        saveAnnotationsToFile(); // Save current annotations before moving
        displayImage(currentImageIndex + 1);
        updateAnnotationsList(); // Update the list for new image
    }
}

function showPreviousImage() {
    if (currentImageIndex > 0) {
        saveAnnotationsToFile(); // Save current annotations before moving
        displayImage(currentImageIndex - 1);
        updateAnnotationsList(); // Update the list for new image
    }
}

async function saveAnnotationsToFile() {
    if (!images[currentImageIndex]) return;
    
    const imageFileName = images[currentImageIndex].name;
    const annotations = images[currentImageIndex].annotations || [];
    
    // Convert annotations to YOLO format
    const yoloAnnotations = annotations.map(ann => ({
        class_name: ann.class,
        coordinates: `${ann.x} ${ann.y} ${ann.width} ${ann.height}`
    }));

    try {
        const response = await fetch('/save_annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_file: imageFileName,
                annotations: yoloAnnotations
            })
        });

        if (!response.ok) {
            throw new Error('Failed to save annotations');
        }

        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || 'Failed to save annotations');
        }

        console.log('Annotations saved successfully');
    } catch (error) {
        console.error('Error saving annotations:', error);
        // Optionally show an error message to the user
        showNotification('error', 'Failed to save annotations');
    }
}

// Add this CSS to your style.css
const style = document.createElement('style');
style.textContent = `
    #canvas-container {
        position: relative;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    }
    
    #canvas {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
`;
document.head.appendChild(style);

// Your existing functions (loadImages, uploadImages, etc.) remain the same...

// Add this to your existing JavaScript
function setupVideoUpload() {
    const videoUpload = document.getElementById('videoUpload');
    videoUpload.addEventListener('change', uploadVideo);
}

async function uploadVideo(file) {
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        const response = await fetch('/upload_video', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (result.success) {
            uploadedVideoFilename = result.filename;
            alert('Video uploaded successfully! Click "Process Video" to extract frames.');
        } else {
            alert('Error uploading video: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error uploading video:', error);
        alert('Error uploading video. Please try again.');
    }
}

async function loadAnnotationsForImage(imageFileName) {
    try {
        const response = await fetch(`/get_annotations/${imageFileName}`);
        if (response.ok) {
            const data = await response.json();
            return data.annotations;
        }
    } catch (error) {
        console.error('Error loading annotations:', error);
    }
    return [];
}

async function startTraining() {
    try {
        // Save current annotations first
        await saveAnnotationsToFile();
        
        // Show loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'trainingProgress';
        loadingDiv.innerHTML = 'Preparing training data...';
        document.body.appendChild(loadingDiv);
        
        const response = await fetch('/prepare_training', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            alert(result.message);
        } else {
            throw new Error(result.error || 'Failed to prepare training data');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error preparing training data: ' + error.message);
    } finally {
        // Remove loading indicator
        const loadingDiv = document.getElementById('trainingProgress');
        if (loadingDiv) {
            document.body.removeChild(loadingDiv);
        }
    }
}

async function clearAnnotations() {
    if (confirm('Are you sure you want to clear ALL annotations for ALL images? This cannot be undone.')) {
        try {
            // Show loading state
            const clearBtn = document.querySelector('.clear-btn');
            const originalText = clearBtn.innerHTML;
            clearBtn.innerHTML = 'Clearing...';
            clearBtn.disabled = true;

            // Clear annotations
            images.forEach(image => {
                image.annotations = [];
            });

            // Clear display
            drawAnnotations();
            updateAnnotationsList();

            // Clear files
            await clearAllAnnotationFiles();

            // Show success message
            showNotification('All annotations cleared successfully', 'success');

        } catch (error) {
            console.error('Error clearing annotations:', error);
            showNotification('Failed to clear annotations', 'error');
        } finally {
            // Reset button
            const clearBtn = document.querySelector('.clear-btn');
            clearBtn.innerHTML = originalText;
            clearBtn.disabled = false;
        }
    }
}

function showNotification(type, message) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    // Add styles for the notification
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.padding = '10px 20px';
    notification.style.borderRadius = '4px';
    notification.style.zIndex = '1000';
    
    if (type === 'error') {
        notification.style.backgroundColor = '#ff4444';
        notification.style.color = 'white';
    } else {
        notification.style.backgroundColor = '#44ff44';
        notification.style.color = 'white';
    }

    // Remove the notification after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

async function clearAllAnnotationFiles() {
    try {
        const response = await fetch('/clear_all_annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            throw new Error('Failed to clear annotations');
        }

        const result = await response.json();
        if (result.success) {
            console.log('All annotations cleared successfully');
        }
    } catch (error) {
        console.error('Error clearing annotations:', error);
        throw error;
    }
}

// Add this function to populate existing classes
function updateClassList() {
    const datalist = document.getElementById('existingClasses');
    if (!datalist) {
        console.error('Datalist element not found');
        return;
    }
    
    // Get unique class names from all annotations
    const existingClasses = new Set();
    
    images.forEach(img => {
        if (img.annotations) {
            img.annotations.forEach(ann => {
                if (ann.class) {
                    existingClasses.add(ann.class);
                }
            });
        }
    });
    
    // Clear and update datalist
    datalist.innerHTML = '';
    existingClasses.forEach(className => {
        const option = document.createElement('option');
        option.value = className;
        datalist.appendChild(option);
    });
}

// Add keyboard handling for the popup
document.getElementById('className').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        const className = this.value.trim();
        if (className) {
            saveAnnotation(className);
        }
    }
});

// Close popup when clicking outside
document.getElementById('classPopup').addEventListener('click', function(e) {
    if (e.target === this) {
        hideClassPopup();
    }
});

function updateAnnotationsList() {
    const tbody = document.getElementById('annotationsList');
    tbody.innerHTML = ''; // Clear existing list
    
    if (!images[currentImageIndex] || !images[currentImageIndex].annotations) {
        return;
    }
    
    images[currentImageIndex].annotations.forEach((ann, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${ann.class}</td>
            <td>
                x: ${ann.x.toFixed(3)}<br>
                y: ${ann.y.toFixed(3)}<br>
                w: ${ann.width.toFixed(3)}<br>
                h: ${ann.height.toFixed(3)}
            </td>
            <td>
                <button class="action-btn edit-btn" onclick="editAnnotation(${index})">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="action-btn delete-btn" onclick="deleteAnnotation(${index})">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });

    // Update image counter
    updateImageCounter();
}

// Update the button style to make it more prominent for a dangerous action
document.querySelector('.clear-btn').style.cssText = `
    background-color: #dc3545;
    color: white;
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
    transition: all 0.3s ease;
`;

// Add this to your existing JavaScript
document.getElementById('videoUpload').addEventListener('change', function(e) {
    const videoUploadControls = document.getElementById('videoUploadControls');
    if (e.target.files.length > 0) {
        selectedVideoFile = e.target.files[0];
        videoUploadControls.style.display = 'block';
        // Upload the video immediately
        uploadVideo(selectedVideoFile);
    } else {
        selectedVideoFile = null;
        uploadedVideoFilename = null;
        videoUploadControls.style.display = 'none';
    }
});

document.getElementById('processVideoBtn').addEventListener('click', async function() {
    if (!uploadedVideoFilename) {
        alert('Please upload a video first');
        return;
    }

    const frameCount = document.getElementById('frameCount').value;
    if (frameCount < 1) {
        alert('Please enter a valid number of frames');
        return;
    }

    // Show loading state
    const btn = this;
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = 'Processing...';

    try {
        const response = await fetch('/process_video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: uploadedVideoFilename,
                frame_count: parseInt(frameCount)
            })
        });

        const result = await response.json();
        
        if (result.success) {
            alert(`Video processed successfully! Extracted ${result.frames_extracted} frames.`);
            // Optionally refresh the image gallery or update UI
            loadImages(); // Assuming you have this function to refresh images
        } else {
            alert('Error: ' + (result.error || 'Failed to process video'));
        }
    } catch (error) {
        console.error('Error processing video:', error);
        alert('Error processing video. Please try again.');
    } finally {
        // Reset button state
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
});

// Add these event listeners
document.getElementById('makeDatasetBtn').addEventListener('click', async function() {
    const btn = this;
    try {
        // Show loading state
        btn.disabled = true;
        btn.classList.add('btn-loading');
        
        const response = await fetch('/prepare_training', {
            method: 'POST'
        });
        const result = await response.json();
        
        if (result.success) {
            showNotification('success', result.message);
        } else {
            showNotification('error', 'Error: ' + (result.error || 'Failed to prepare dataset'));
        }
    } catch (error) {
        console.error('Error preparing dataset:', error);
        showNotification('error', 'Error preparing dataset. Please try again.');
    } finally {
        // Reset button state
        btn.disabled = false;
        btn.classList.remove('btn-loading');
    }
});

document.getElementById('startTrainingBtn').addEventListener('click', function() {
    const trainingPanel = document.getElementById('trainingPanel');
    if (trainingPanel.style.display === 'none') {
        trainingPanel.style.display = 'block';
        trainingPanel.scrollIntoView({ behavior: 'smooth' });
    } else {
        trainingPanel.style.display = 'none';
    }
});

document.getElementById('executeTrainingBtn').addEventListener('click', async function() {
    const epochs = document.getElementById('epochCount').value;
    const btn = this;
    
    if (!epochs || epochs < 1) {
        showNotification('error', 'Please enter a valid number of epochs');
        return;
    }
    
    try {
        // Show loading state
        btn.disabled = true;
        btn.classList.add('btn-loading');
        
        const response = await fetch('/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                epochs: parseInt(epochs)
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('success', 'Training started successfully!');
            document.getElementById('trainingPanel').style.display = 'none';
        } else {
            showNotification('error', 'Error: ' + (result.error || 'Failed to start training'));
        }
    } catch (error) {
        console.error('Error starting training:', error);
        showNotification('error', 'Error starting training. Please try again.');
    } finally {
        // Reset button state
        btn.disabled = false;
        btn.classList.remove('btn-loading');
    }
});

// Add these event listeners to your existing JavaScript
document.getElementById('closeTrainingPanel').addEventListener('click', function() {
    document.getElementById('trainingPanel').style.display = 'none';
});

// Add this helper function for notifications
function showNotification(type, message) {
    // You can implement this using your preferred notification library
    // For now, we'll use alert
    alert(message);
}

// Make sure popup is hidden when page loads
document.addEventListener('DOMContentLoaded', function() {
    hideClassPopup();
});

// Update your existing event listeners
document.getElementById('cancelAnnotation').addEventListener('click', hideClassPopup);

// When you need to show the popup (e.g., after drawing a box)
// Call showClassPopup() instead of directly manipulating style

// Helper function to draw annotation boxes
function drawAnnotationBox(ctx, annotation) {
    const { x, y, width, height, class: className } = annotation;
    
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    
    // Convert normalized coordinates to canvas coordinates
    const canvasX = x * canvas.width;
    const canvasY = y * canvas.height;
    const canvasWidth = width * canvas.width;
    const canvasHeight = height * canvas.height;
    
    ctx.strokeRect(canvasX, canvasY, canvasWidth, canvasHeight);
    
    // Draw label
    ctx.fillStyle = '#00ff00';
    ctx.font = '14px Arial';
    const padding = 2;
    ctx.fillText(className, canvasX, canvasY - padding);
}

// Update the mouse event handlers
function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (evt.clientX - rect.left) / canvasScale.x,
        y: (evt.clientY - rect.top) / canvasScale.y
    };
}

// Update the mousedown event handler
canvas.addEventListener('mousedown', function(e) {
    if (isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    startX = (e.clientX - rect.left) * scaleX;
    startY = (e.clientY - rect.top) * scaleY;
    
    isDrawing = true;
});

// Update the mousemove event handler
canvas.addEventListener('mousemove', function(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const currentX = (e.clientX - rect.left) * scaleX;
    const currentY = (e.clientY - rect.top) * scaleY;
    
    // Redraw the image and all annotations
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    
    // Draw existing annotations
    if (images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations.forEach(annotation => {
            drawAnnotationBox(ctx, annotation);
        });
    }
    
    // Draw current rectangle
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.strokeRect(
        startX,
        startY,
        currentX - startX,
        currentY - startY
    );
});

// Update the mouseup event handler
canvas.addEventListener('mouseup', function(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const endX = (e.clientX - rect.left) * scaleX;
    const endY = (e.clientY - rect.top) * scaleY;
    
    // Calculate normalized coordinates
    const x = Math.min(startX, endX) / canvas.width;
    const y = Math.min(startY, endY) / canvas.height;
    const width = Math.abs(endX - startX) / canvas.width;
    const height = Math.abs(endY - startY) / canvas.height;
    
    // Only show class popup if the box has some size
    if (width > 0.01 && height > 0.01) {
        currentAnnotation = { x, y, width, height };
        showClassPopup();
    }
    
    isDrawing = false;
});

// Update the addAnnotation function
function addAnnotation(className) {
    if (!currentAnnotation || !className || typeof className !== 'string') {
        console.error('Invalid annotation or class name');
        return;
    }

    const canvas = document.getElementById('canvas');
    
    // Create normalized annotation
    const annotation = {
        x: currentAnnotation.x / canvas.width,
        y: currentAnnotation.y / canvas.height,
        width: currentAnnotation.width / canvas.width,
        height: currentAnnotation.height / canvas.height,
        class: className
    };

    // Initialize annotations array if needed
    if (!images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations = [];
    }

    // Add the annotation
    images[currentImageIndex].annotations.push(annotation);

    // Update display
    drawAnnotations();
    updateAnnotationsList();
    saveAnnotationsToFile();

    // Reset current annotation
    currentAnnotation = null;
}

// Update the confirm class button handler
document.getElementById('confirmClass').addEventListener('click', function() {
    const className = document.getElementById('className').value.trim();
    console.log('Confirming class:', className);
    if (className && currentAnnotation) {
        saveAnnotation(className);
    } else {
        console.error('Missing className or currentAnnotation:', { className, currentAnnotation });
    }
});

// Update addAnnotation function
function addAnnotation(className) {
    if (!currentAnnotation) return;
    
    const annotation = {
        ...currentAnnotation,
        class: className  // Use the passed className string
    };
    
    if (!images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations = [];
    }
    
    images[currentImageIndex].annotations.push(annotation);
    
    // Redraw everything
    drawAnnotations();
    updateAnnotationsList();
    saveAnnotationsToFile();
    
    currentAnnotation = null;
}

// Update the class list function
function updateClassList() {
    const datalist = document.getElementById('existingClasses');
    if (!datalist) {
        console.error('Datalist element not found');
        return;
    }
    
    // Get unique class names from all annotations
    const existingClasses = new Set();
    
    images.forEach(img => {
        if (img.annotations) {
            img.annotations.forEach(ann => {
                if (ann.class) {
                    existingClasses.add(ann.class);
                }
            });
        }
    });
    
    // Clear and update datalist
    datalist.innerHTML = '';
    existingClasses.forEach(className => {
        const option = document.createElement('option');
        option.value = className;
        datalist.appendChild(option);
    });
}

// Update the event listeners
document.addEventListener('DOMContentLoaded', function() {
    const confirmButton = document.getElementById('confirmClass');
    const cancelButton = document.getElementById('cancelAnnotation');
    const classInput = document.getElementById('className');

    // Confirm button handler
    if (confirmButton) {
        confirmButton.addEventListener('click', function(e) {
            e.preventDefault();
            const inputValue = classInput.value.trim();
            console.log('Confirm clicked with value:', inputValue); // Debug log
            
            if (inputValue) {
                handleAnnotation(inputValue);
            }
        });
    }

    // Cancel button handler
    if (cancelButton) {
        cancelButton.addEventListener('click', function() {
            currentAnnotation = null;
            hideClassPopup();
            drawAnnotations();
        });
    }

    // Enter key handler
    if (classInput) {
        classInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                const inputValue = this.value.trim();
                console.log('Enter pressed with value:', inputValue); // Debug log
                
                if (inputValue) {
                    handleAnnotation(inputValue);
                }
            }
        });
    }
});

// Update the annotation handling function
function handleAnnotation(className) {
    console.log('Handling annotation with:', className); // Debug log
    
    if (!currentAnnotation) {
        console.error('No current annotation exists');
        return;
    }

    const canvas = document.getElementById('canvas');
    
    // Create new annotation object
    const newAnnotation = {
        x: currentAnnotation.x / canvas.width,
        y: currentAnnotation.y / canvas.height,
        width: currentAnnotation.width / canvas.width,
        height: currentAnnotation.height / canvas.height,
        class: className
    };

    console.log('Created annotation:', newAnnotation); // Debug log

    // Add to annotations array
    if (!images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations = [];
    }
    images[currentImageIndex].annotations.push(newAnnotation);

    // Update UI
    hideClassPopup();
    drawAnnotations();
    updateAnnotationsList();
    saveAnnotationsToFile();
    currentAnnotation = null;
}

// Update showClassPopup function
function showClassPopup() {
    const popup = document.getElementById('classPopup');
    const classInput = document.getElementById('className');
    
    if (!popup || !classInput) {
        console.error('Required popup elements not found');
        return;
    }
    
    popup.style.display = 'flex';
    classInput.value = '';
    classInput.focus();
}

// Update hideClassPopup function
function hideClassPopup() {
    const popup = document.getElementById('classPopup');
    const classInput = document.getElementById('className');
    
    if (popup) {
        popup.style.display = 'none';
    }
    if (classInput) {
        classInput.value = '';
    }
}