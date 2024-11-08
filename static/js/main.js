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

let notificationQueue = [];
let isNotificationDisplaying = false;

function initializeApp() {
    console.log("Initializing app...");
    loadImages();
    
    const canvas = document.getElementById('canvas');
    const confirmClass = document.getElementById('confirmClass');
    const cancelAnnotation = document.getElementById('cancelAnnotation');
    const clearAnnotationsBtn = document.getElementById('clearAnnotations');
    
    console.log("Clear Annotations Button:", clearAnnotationsBtn);
    
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
    
    // Clear buttons with debug logs
    document.getElementById('clearImages').addEventListener('click', () => {
        console.log("Clear Images clicked");
        clearImages();
    });
    
    if (clearAnnotationsBtn) {
        clearAnnotationsBtn.addEventListener('click', () => {
            console.log("Clear Annotations clicked");
            clearAllAnnotations();
        });
    } else {
        console.error("Clear Annotations button not found!");
    }
    
    // Training buttons
    document.getElementById('startProcess').addEventListener('click', startProcess);

    // Add video details button listener
    document.getElementById('showDetailsBtn').addEventListener('click', () => {
        console.log("Show Details clicked");
        if (uploadedVideoFilename) {
            showVideoDetails(uploadedVideoFilename);
        } else {
            alert('Please upload a video first');
        }
    });

    // Test section controls
    document.getElementById('testImage').addEventListener('change', uploadTestFiles);
    document.getElementById('testVideo').addEventListener('change', uploadTestFiles);
    document.getElementById('startTesting').addEventListener('click', startTesting);

    // Add download results button listener
    document.getElementById('downloadResults').addEventListener('click', downloadResults);

    // Add this to your existing initializeApp function
    document.getElementById('deleteProject').addEventListener('click', deleteProject);
}

async function uploadImages(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;

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
            // Reload images after successful upload
            await loadImages();
            // Reset the file input
            e.target.value = '';
        } else {
            console.error('Upload failed');
        }
    } catch (error) {
        console.error('Error uploading files:', error);
    }
}

async function loadImages() {
    try {
        const response = await fetch('/get_images');
        const data = await response.json();
        images = data.images;
        updateImageCounter();
        if (images.length > 0) {
            displayImage(currentImageIndex);
        }
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
    console.log("Start drawing");
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = (e.clientX - rect.left) / canvas.width;
    startY = (e.clientY - rect.top) / canvas.height;
}

function draw(e) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const currentX = (e.clientX - rect.left) / canvas.width;
    const currentY = (e.clientY - rect.top) / canvas.height;

    // Clear previous drawings
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Redraw image
    if (currentImage) {
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    }

    // Draw existing annotations
    if (images[currentImageIndex] && images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations.forEach(ann => {
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.strokeRect(
                ann.x * canvas.width,
                ann.y * canvas.height,
                ann.width * canvas.width,
                ann.height * canvas.height
            );
        });
    }

    // Draw current rectangle
    ctx.strokeStyle = '#ff0000';  // Red color for current drawing
    ctx.lineWidth = 2;
    const width = currentX - startX;
    const height = currentY - startY;
    ctx.strokeRect(
        startX * canvas.width,
        startY * canvas.height,
        width * canvas.width,
        height * canvas.height
    );
}

function endDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;

    const rect = canvas.getBoundingClientRect();
    const endX = (e.clientX - rect.left) / canvas.width;
    const endY = (e.clientY - rect.top) / canvas.height;

    // Calculate dimensions
    const width = Math.abs(endX - startX);
    const height = Math.abs(endY - startY);
    const x = Math.min(startX, endX);
    const y = Math.min(startY, endY);

    // Only create annotation if the box has some size
    if (width > 0.01 && height > 0.01) {
        currentAnnotation = {
            x: x,
            y: y,
            width: width,
            height: height
        };

        // Show class input popup
        const popup = document.getElementById('classPopup');
        popup.style.display = 'block';
        document.getElementById('className').value = '';
        document.getElementById('className').focus();
    }
}

function showClassPopup() {
    const popup = document.getElementById('classPopup');
    const className = document.getElementById('className');
    updateClassList(); // Update the list of existing classes
    popup.style.display = 'flex';
    className.value = '';
    className.focus();
}

async function saveAnnotation() {
    try {
        const className = document.getElementById('className').value;
        if (!className) {
            console.log("No class name provided");
            return;
        }

        if (currentAnnotation) {
            // Convert to YOLO format (center coordinates)
            const x_center = currentAnnotation.x + (currentAnnotation.width / 2);
            const y_center = currentAnnotation.y + (currentAnnotation.height / 2);
            
            const annotation = {
                x: x_center,          // Center X instead of top-left X
                y: y_center,          // Center Y instead of top-left Y
                width: currentAnnotation.width,
                height: currentAnnotation.height,
                class: className
            };

            if (!images[currentImageIndex].annotations) {
                images[currentImageIndex].annotations = [];
            }

            images[currentImageIndex].annotations.push(annotation);
            
            // Save to file
            saveAnnotationsToFile();
            
            document.getElementById('classPopup').style.display = 'none';
            currentAnnotation = null;
            
            drawAnnotations();
            updateAnnotationsList();
        }
    } catch (error) {
        console.error('Error in saveAnnotation:', error);
        alert('Error saving annotation: ' + error.message);
    }
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
        width = box.width * canvasWidth;
        height = box.height * canvasHeight;
        x = (box.x * canvasWidth) - (width / 2);
        y = (box.y * canvasHeight) - (height / 2);
    } else {
        x = box.x;
        y = box.y;
        width = box.width;
        height = box.height;
    }

    // Draw rectangle with gradient stroke
    ctx.beginPath();
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#2196F3';
    ctx.setLineDash([]);
    ctx.rect(x, y, width, height);
    ctx.stroke();

    // Draw class name background
    if (className) {
        ctx.font = '14px Inter';
        const textWidth = ctx.measureText(className).width;
        const textHeight = 20;
        const padding = 8;
        
        ctx.fillStyle = '#2196F3';
        ctx.fillRect(x - 1, y - textHeight - padding, textWidth + padding * 2, textHeight + padding);
        
        // Draw class name text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(className, x + padding, y - padding);
    }

    // Draw corner markers
    const markerSize = 8;
    ctx.fillStyle = '#2196F3';
    
    // Top-left
    ctx.fillRect(x - markerSize/2, y - markerSize/2, markerSize, markerSize);
    // Top-right
    ctx.fillRect(x + width - markerSize/2, y - markerSize/2, markerSize, markerSize);
    // Bottom-left
    ctx.fillRect(x - markerSize/2, y + height - markerSize/2, markerSize, markerSize);
    // Bottom-right
    ctx.fillRect(x + width - markerSize/2, y + height - markerSize/2, markerSize, markerSize);
}

function drawAnnotations() {
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw image
    if (currentImage) {
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    }

    // Draw all annotations
    if (images[currentImageIndex] && images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations.forEach(ann => {
            ctx.strokeStyle = '#00ff00';  // Green color
            ctx.lineWidth = 2;
            
            // Convert from center to corner coordinates for drawing
            const x = (ann.x - ann.width/2) * canvas.width;
            const y = (ann.y - ann.height/2) * canvas.height;
            const w = ann.width * canvas.width;
            const h = ann.height * canvas.height;
            
            ctx.strokeRect(x, y, w, h);

            // Draw label
            ctx.fillStyle = '#00ff00';
            ctx.font = '12px Arial';
            ctx.fillText(ann.class, x, y - 5);
        });
    }
}

async function displayImage(index) {
    if (images.length === 0 || index < 0 || index >= images.length) return;
    
    const annotations = await loadAnnotationsForImage(images[index].name);
    images[index].annotations = annotations;
    
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('canvas-container');
    
    const img = new Image();
    img.onload = function() {
        currentImage = img;
        
        // Determine if image is horizontal or vertical
        const isHorizontal = img.width > img.height;
        
        if (isHorizontal) {
            canvas.width = 900;
            canvas.height = 560;
            container.style.width = '900px';
            container.style.height = '560px';
        } else {
            canvas.width = 560;
            canvas.height = 900;
            container.style.width = '560px';
            container.style.height = '900px';
        }
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const scale = Math.min(
            (canvas.width * 0.95) / img.width,
            (canvas.height * 0.95) / img.height
        );
        
        const x = (canvas.width - img.width * scale) / 2;
        const y = (canvas.height - img.height * scale) / 2;
        
        ctx.drawImage(
            img,
            x, y,
            img.width * scale,
            img.height * scale
        );
        
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
    const yoloAnnotations = annotations.map(ann => {
        return {
            class_name: ann.class,
            coordinates: `${ann.x} ${ann.y} ${ann.width} ${ann.height}`
        };
    });

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
    } catch (error) {
        console.error('Error saving annotations:', error);
        alert('Failed to save annotations');
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
    videoUpload.addEventListener('change', function(e) {
        const videoUploadControls = document.getElementById('videoUploadControls');
        if (e.target.files.length > 0) {
            selectedVideoFile = e.target.files[0];
            videoUploadControls.style.display = 'block';
            uploadVideo(selectedVideoFile);
        } else {
            selectedVideoFile = null;
            uploadedVideoFilename = null;
            videoUploadControls.style.display = 'none';
        }
    });

    // Add event listener for Show Details button
    document.getElementById('showDetailsBtn').addEventListener('click', async function() {
        if (uploadedVideoFilename) {
            try {
                const response = await fetch('/get_video_details', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        filename: uploadedVideoFilename
                    })
                });

                const result = await response.json();
                if (result.success) {
                    const detailsHtml = `
                        <div class="video-details">
                            <p><strong>Duration:</strong> ${result.duration.toFixed(2)} seconds</p>
                            <p><strong>FPS:</strong> ${result.fps.toFixed(2)}</p>
                            <p><strong>Total Frames:</strong> ${result.total_frames}</p>
                        </div>
                    `;
                    document.getElementById('videoDetails').innerHTML = detailsHtml;
                    document.getElementById('videoDetails').style.display = 'block';
                } else {
                    alert('Error getting video details: ' + result.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error getting video details');
            }
        } else {
            alert('Please upload a video first');
        }
    });
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

async function startProcess() {
    try {
        // First, create the database
        showNotification('Creating dataset...', 'info');
        
        const prepareResponse = await fetch('/prepare_training', {
            method: 'POST'
        });
        
        if (!prepareResponse.ok) {
            throw new Error('Failed to prepare training data');
        }
        
        const prepareResult = await prepareResponse.json();
        if (!prepareResult.success) {
            throw new Error(prepareResult.error || 'Dataset creation failed');
        }
        
        showNotification('Dataset created, starting training...', 'info');
        
        // Then start the training
        const trainingResponse = await fetch('/training_progress');
        
        if (!trainingResponse.ok) {
            throw new Error('Training failed');
        }
        
        const trainingResult = await trainingResponse.json();
        if (trainingResult.success) {
            showNotification('Training completed successfully!', 'success');
        } else {
            throw new Error(trainingResult.error || 'Training failed');
        }
        
    } catch (error) {
        console.error('Error in process:', error);
        showNotification('Error: ' + error.message, 'error');
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

function showNotification(message, type = 'info') {
    // Add notification to queue
    notificationQueue.push({ message, type });
    
    // If no notification is currently showing, display the next one
    if (!isNotificationDisplaying) {
        displayNextNotification();
    }
}

function displayNextNotification() {
    if (notificationQueue.length === 0) {
        isNotificationDisplaying = false;
        return;
    }

    isNotificationDisplaying = true;
    const { message, type } = notificationQueue.shift();

    // Remove any existing notifications
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }

    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    // Add close button
    const closeButton = document.createElement('span');
    closeButton.className = 'notification-close';
    closeButton.innerHTML = '×';
    closeButton.onclick = () => {
        notification.remove();
        isNotificationDisplaying = false;
        displayNextNotification();
    };

    // Add message
    const messageSpan = document.createElement('span');
    messageSpan.textContent = message;

    notification.appendChild(messageSpan);
    notification.appendChild(closeButton);
    document.body.appendChild(notification);

    // Auto remove after 3 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
            isNotificationDisplaying = false;
            displayNextNotification();
        }
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
    const existingClasses = new Set();
    
    // Collect all existing class names
    images.forEach(img => {
        if (img.annotations) {
            img.annotations.forEach(ann => {
                if (ann.class) {
                    existingClasses.add(ann.class);
                }
            });
        }
    });
    
    // Update datalist
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
        document.getElementById('confirmClass').click();
    }
});

// Close popup when clicking outside
document.getElementById('classPopup').addEventListener('click', function(e) {
    if (e.target === this) {
        document.getElementById('cancelAnnotation').click();
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

function clearImages() {
    console.log("clearImages function called");
    if (confirm('Are you sure you want to delete all images and their annotations? This cannot be undone.')) {
        console.log("User confirmed image deletion");
        fetch('/clear_images', {
            method: 'POST',
        })
        .then(response => {
            console.log("Clear images response:", response);
            return response.json();
        })
        .then(data => {
            console.log("Clear images data:", data);
            if (data.success) {
                return fetch('/clear_all_annotations', {
                    method: 'POST'
                });
            }
            throw new Error(data.error || 'Failed to clear images');
        })
        .then(response => response.json())
        .then(data => {
            console.log("Clear annotations data:", data);
            if (data.success) {
                // Reset the UI
                images = [];
                currentImageIndex = 0;
                currentImage = null;
                
                // Clear canvas
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Clear annotations list
                const annotationsList = document.getElementById('annotationsList');
                if (annotationsList) {
                    annotationsList.innerHTML = '';
                }
                
                // Update UI elements
                updateImageCounter();
                
                alert('Images and annotations cleared successfully');
                loadImages();
            }
        })
        .catch(error => {
            console.error('Error in clearImages:', error);
            alert('Error clearing images and annotations');
        });
    } else {
        console.log("User cancelled image deletion");
    }
}

// Update the video processing to delete the video after extraction
async function processVideo(filename, frameCount) {
    try {
        const response = await fetch('/process_video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: filename,
                frame_count: parseInt(frameCount)
            })
        });

        const result = await response.json();
        
        if (result.success) {
            // Reset video upload related variables
            selectedVideoFile = null;
            uploadedVideoFilename = null;
            document.getElementById('videoUpload').value = '';
            document.getElementById('videoUploadControls').style.display = 'none';
            
            // Refresh images
            await loadImages();
            
            showNotification(`Extracted ${result.frames_extracted} frames successfully`, 'success');
        } else {
            throw new Error(result.error || 'Failed to process video');
        }
    } catch (error) {
        console.error('Error processing video:', error);
        showNotification('Error processing video', 'error');
        throw error;
    }
}

// Add the clearAllAnnotations function
function clearAllAnnotations() {
    console.log("clearAllAnnotations function called");
    if (confirm('Are you sure you want to delete all annotations? This cannot be undone.')) {
        console.log("User confirmed deletion");
        fetch('/clear_all_annotations', {
            method: 'POST'
        })
        .then(response => {
            console.log("Response received:", response);
            return response.json();
        })
        .then(data => {
            console.log("Data received:", data);
            if (data.success) {
                console.log("Clearing annotations from UI");
                // Clear annotations from current display
                if (images[currentImageIndex]) {
                    images[currentImageIndex].annotations = [];
                }
                
                // Clear annotations list
                const annotationsList = document.getElementById('annotationsList');
                if (annotationsList) {
                    annotationsList.innerHTML = '';
                }
                
                // Redraw canvas
                drawAnnotations();
                
                alert('All annotations cleared successfully');
            } else {
                throw new Error(data.error || 'Failed to clear annotations');
            }
        })
        .catch(error => {
            console.error('Error in clearAllAnnotations:', error);
            alert('Error clearing annotations');
        });
    } else {
        console.log("User cancelled deletion");
    }
}

// Add the showVideoDetails function
async function showVideoDetails(filename) {
    try {
        console.log("Fetching video details for:", filename);
        const response = await fetch('/get_video_details', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filename: filename })
        });

        const data = await response.json();
        console.log("Video details received:", data);

        if (data.success) {
            const detailsDiv = document.getElementById('videoDetails');
            detailsDiv.innerHTML = `
                <div style="margin-top: 10px; padding: 10px; background: #f5f5f5; border-radius: 5px;">
                    <p><strong>FPS:</strong> ${data.fps.toFixed(2)}</p>
                    <p><strong>Total Frames:</strong> ${data.total_frames}</p>
                    <p><strong>Duration:</strong> ${data.duration.toFixed(2)} seconds</p>
                </div>
            `;
            detailsDiv.style.display = 'block';
        } else {
            throw new Error(data.error || 'Failed to get video details');
        }
    } catch (error) {
        console.error('Error getting video details:', error);
        alert('Error getting video details: ' + error.message);
    }
}

// Make sure uploadedVideoFilename is set when a video is uploaded
document.getElementById('videoUpload').addEventListener('change', async function(e) {
    if (this.files && this.files[0]) {
        const formData = new FormData();
        formData.append('video', this.files[0]);
        
        try {
            const response = await fetch('/upload_video', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            if (data.success) {
                uploadedVideoFilename = data.filename;  // Save the filename
                document.getElementById('videoUploadControls').style.display = 'block';
                console.log("Video uploaded:", uploadedVideoFilename);
            } else {
                throw new Error(data.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error uploading video: ' + error.message);
        }
    }
});

async function uploadTestFiles(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    console.log("Starting test file upload...");  // Debug log
    
    const formData = new FormData();
    for (let file of files) {
        formData.append('files[]', file);
        console.log("Adding file to upload:", file.name);  // Debug log
    }
    
    try {
        showNotification('Uploading test files...', 'info');
        
        const response = await fetch('/upload_test_files', {
            method: 'POST',
            body: formData
        });
        
        console.log("Upload response status:", response.status);  // Debug log
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log("Upload result:", result);  // Debug log
        
        showNotification(`Successfully uploaded ${result.files.length} files`, 'success');
        
        // Reset the file input
        e.target.value = '';
        
    } catch (error) {
        console.error('Error uploading test files:', error);
        showNotification('Error uploading test files: ' + error.message, 'error');
    }
}

// Make sure the event listeners are properly set up
document.addEventListener('DOMContentLoaded', function() {
    console.log("Setting up test file upload listeners");  // Debug log
    
    const testImage = document.getElementById('testImage');
    const testVideo = document.getElementById('testVideo');
    
    if (testImage) {
        testImage.addEventListener('change', uploadTestFiles);
        console.log("Test image upload listener added");  // Debug log
    }
    
    if (testVideo) {
        testVideo.addEventListener('change', uploadTestFiles);
        console.log("Test video upload listener added");  // Debug log
    }
});

async function startTesting() {
    try {
        const downloadBtn = document.getElementById('downloadResults');
        downloadBtn.disabled = true;
        showNotification('Starting test...', 'info');
        
        const response = await fetch('/start_testing', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            showNotification('Testing completed successfully!', 'success');
            downloadBtn.disabled = false;
        } else {
            const message = result.message || result.error || 'Testing failed to start';
            showNotification(message, result.success ? 'info' : 'error');
            if (!result.success) {
                downloadBtn.disabled = true;
            }
        }
    } catch (error) {
        console.error('Error starting testing:', error);
        showNotification('Error during testing: ' + error.message, 'error');
        downloadBtn.disabled = true;
    }
}

async function downloadResults() {
    try {
        showNotification('Preparing download...', 'info');
        
        const response = await fetch('/download_all_results');
        
        if (!response.ok) {
            throw new Error('Failed to download results');
        }
        
        // Get the filename from the Content-Disposition header
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'all_results.zip';
        if (contentDisposition) {
            const match = contentDisposition.match(/filename="(.+)"/);
            if (match) {
                filename = match[1];
            }
        }
        
        // Download the zip file
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showNotification('Results downloaded successfully!', 'success');
    } catch (error) {
        console.error('Error downloading results:', error);
        showNotification('Error downloading results: ' + error.message, 'error');
    }
}

// Add this function
async function deleteProject() {
    if (confirm('Are you sure you want to delete all project data? This cannot be undone.')) {
        try {
            showNotification('Deleting project data...', 'info');
            
            const response = await fetch('/delete_project', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                showNotification('Project deleted successfully', 'success');
                location.reload(); // Refresh the page to reset all states
            } else {
                throw new Error(result.error || 'Failed to delete project');
            }
        } catch (error) {
            console.error('Error deleting project:', error);
            showNotification('Error deleting project: ' + error.message, 'error');
        }
    }
}