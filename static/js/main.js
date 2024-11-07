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
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    currentAnnotation = { x: startX, y: startY, width: 0, height: 0 };
    drawAnnotations();
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    currentAnnotation.width = currentX - startX;
    currentAnnotation.height = currentY - startY;
    
    drawAnnotations();
}

function endDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;

    if (currentAnnotation && Math.abs(currentAnnotation.width) > 5 && Math.abs(currentAnnotation.height) > 5) {
        showClassPopup();
    } else {
        currentAnnotation = null;
        drawAnnotations();
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

function saveAnnotation() {
    const className = document.getElementById('className').value.trim();
    if (!className || !currentAnnotation) return;

    const canvas = document.getElementById('canvas');
    
    // Convert to YOLO format (normalized coordinates)
    const x = Math.min(currentAnnotation.x, currentAnnotation.x + currentAnnotation.width);
    const y = Math.min(currentAnnotation.y, currentAnnotation.y + currentAnnotation.height);
    const width = Math.abs(currentAnnotation.width);
    const height = Math.abs(currentAnnotation.height);

    const annotation = {
        x: (x + width/2) / canvas.width,
        y: (y + height/2) / canvas.height,
        width: width / canvas.width,
        height: height / canvas.height,
        class: className
    };

    if (!images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations = [];
    }
    images[currentImageIndex].annotations.push(annotation);

    // Hide popup and reset current annotation
    document.getElementById('classPopup').style.display = 'none';
    currentAnnotation = null;
    
    // Update display
    updateAnnotationsList();
    drawAnnotations();
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
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    // Clear and redraw image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (currentImage) {
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    }

    // Draw all saved annotations
    if (images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations.forEach(ann => {
            drawBox(ctx, { ...ann, normalized: true }, ann.class);
        });
    }

    // Draw current annotation if drawing
    if (currentAnnotation) {
        drawBox(ctx, currentAnnotation);
    }
}

async function displayImage(index) {
    if (images.length === 0 || index < 0 || index >= images.length) return;
    
    // Load annotations for this image
    const annotations = await loadAnnotationsForImage(images[index].name);
    images[index].annotations = annotations;
    
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const canvasContainer = document.getElementById('canvas-container');
    
    const img = new Image();
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
        
        canvas.width = renderWidth;
        canvas.height = renderHeight;
        
        drawAnnotations();
        updateAnnotationsList(); // Update list after loading image
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

function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

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