# Training Time - 
# Testing Time - 7 mins and 9 secs
import zipfile
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, send_file
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import cv2
from collections import OrderedDict
import json
import shutil
import random
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import pytesseract
import asyncio
import aiofiles
from concurrent.futures import ProcessPoolExecutor
from openpyxl import Workbook
from time import time

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
VIDEO_FOLDER = 'static/videos'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'wmv'}
CONFIDENCE_THRESHOLD = 0.6

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB max-limit

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Add these global variables
LABELS_FILE = 'labels.txt'
ANNOTATIONS_FOLDER = 'annotations'
DATASET_ROOT = 'dataset'

# Create annotations directory
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)

# Initialize or load class labels
def init_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            return OrderedDict((label.strip(), idx) for idx, label in enumerate(f))
    return OrderedDict()

class_labels = init_labels()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files[]')
    uploaded_files = []
    
    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Create empty annotation file
                base_filename = os.path.splitext(filename)[0]
                annotation_path = os.path.join(ANNOTATIONS_FOLDER, f"{base_filename}.txt")
                if not os.path.exists(annotation_path):
                    open(annotation_path, 'a').close()
                
                uploaded_files.append(filename)
        
        return jsonify({'files': uploaded_files, 'success': True})
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_images')
def get_images():
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            files.append({
                'name': filename,
                'url': f'/uploads/{filename}'
            })
    return jsonify({'images': files})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    video = request.files['video']
    
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if video and allowed_file(video.filename):
        try:
            filename = secure_filename(video.filename)
            video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
            video.save(video_path)
            
            return jsonify({
                'success': True,
                'message': 'Video uploaded successfully.',
                'filename': filename
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid video format'}), 400

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        data = request.json
        filename = data.get('filename')
        skip_frames = int(data.get('frame_count', 2))
        
        video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
            
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'])
        extracted_frames = extract_frames(video_path, output_dir, skip_frames=skip_frames)

        # Delete the video file after processing
        try:
            os.remove(video_path)
        except Exception as e:
            print(f"Error deleting video file: {e}")

        return jsonify({
            'success': True,
            'message': f'Video processed successfully. Extracted {extracted_frames} frames by skipping every {skip_frames} frames.',
            'frames_extracted': extracted_frames
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_frames(video_path, output_dir, skip_frames=2):
    """Extract frames from video by skipping specified number of frames"""
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video {video_path}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame if it's at the skip interval
        if frame_count % skip_frames == 0:
            frame_filename = os.path.join(output_dir, f"{video_name}_frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    return saved_count

@app.route('/save_annotations', methods=['POST'])
def save_annotations():
    try:
        data = request.json
        image_file = data['image_file']
        annotations = data['annotations']
        
        # Get base filename without extension
        base_filename = os.path.splitext(image_file)[0]
        
        # Update class labels
        updated = False
        for ann in annotations:
            class_name = ann['class_name']
            if class_name not in class_labels:
                class_labels[class_name] = len(class_labels)
                updated = True
        
        # Save updated labels if necessary
        if updated:
            with open(LABELS_FILE, 'w') as f:
                for label in class_labels:
                    f.write(f"{label}\n")
        
        # Create YOLO format annotations
        yolo_annotations = []
        for ann in annotations:
            class_idx = class_labels[ann['class_name']]
            coordinates = ann['coordinates']
            yolo_annotations.append(f"{class_idx} {coordinates}")
        
        # Save annotations to file
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, f"{base_filename}.txt")
        with open(annotation_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        return jsonify({'success': True, 'message': 'Annotations saved successfully'})
    
    except Exception as e:
        print(f"Error saving annotations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_annotations/<image_file>')
def get_annotations(image_file):
    try:
        base_filename = os.path.splitext(image_file)[0]
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, f"{base_filename}.txt")
        
        annotations = []
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:  # class_idx x y width height
                        class_idx = int(parts[0])
                        # Find class name from index
                        class_name = next((k for k, v in class_labels.items() if v == class_idx), 'unknown')
                        annotations.append({
                            'class': class_name,
                            'x': float(parts[1]),
                            'y': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        })
        
        return jsonify({'annotations': annotations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def add_noise(image):
    # Gaussian Noise
    def gaussian_noise(img):
        mean = 0
        sigma = 25
        noise = np.random.normal(mean, sigma, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Salt and Pepper Noise
    def salt_pepper_noise(img):
        prob = 0.05
        noisy = np.copy(img)
        # Salt
        salt_mask = np.random.random(img.shape) < prob/2
        noisy[salt_mask] = 255
        # Pepper
        pepper_mask = np.random.random(img.shape) < prob/2
        noisy[pepper_mask] = 0
        return noisy

    return [
        gaussian_noise(image),
        salt_pepper_noise(image)
    ]

@app.route('/prepare_training', methods=['POST'])
def prepare_training():
    try:
        # Create dataset structure
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(DATASET_ROOT, split)
            os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)

        # Get all annotation files to find unique classes
        class_set = set()
        class_names = {}
        
        # Read all annotation files to get unique class numbers
        for ann_file in os.listdir(ANNOTATIONS_FOLDER):
            if ann_file.endswith('.txt'):
                with open(os.path.join(ANNOTATIONS_FOLDER, ann_file), 'r') as f:
                    for line in f:
                        class_num = int(line.split()[0])
                        class_set.add(class_num)
        
        # Read labels.txt to get class names
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                class_names = {i: name.strip() for i, name in enumerate(f.readlines())}
        
        # Create data.yaml file
        yaml_content = {
            'path': os.path.abspath(DATASET_ROOT),
            'train': os.path.join('train', 'images'),
            'val': os.path.join('valid', 'images'),
            'test': os.path.join('test', 'images'),
            'nc': len(class_names),
            'names': [class_names[i] for i in range(len(class_names))]
        }

        yaml_path = os.path.join(DATASET_ROOT, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False)

        # Get all images and their annotations
        image_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
        
        # Split data
        train_files, temp_files = train_test_split(image_files, train_size=0.7, random_state=42)
        valid_files, test_files = train_test_split(temp_files, train_size=0.67, random_state=42)

        def copy_files_with_augmentation(files, split):
            saved_count = 0
            for img_file in files:
                # Copy original image
                src_img = os.path.join(app.config['UPLOAD_FOLDER'], img_file)
                base_name = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]

                # Read original image
                img = cv2.imread(src_img)
                if img is None:
                    continue

                # Save original image
                dst_img = os.path.join(DATASET_ROOT, split, 'images', f"{base_name}_orig{ext}")
                cv2.imwrite(dst_img, img)
                saved_count += 1

                # Copy original annotation
                ann_file = f"{base_name}.txt"
                src_ann = os.path.join(ANNOTATIONS_FOLDER, ann_file)
                if os.path.exists(src_ann):
                    dst_ann = os.path.join(DATASET_ROOT, split, 'labels', f"{base_name}_orig.txt")
                    shutil.copy2(src_ann, dst_ann)

                # Generate and save augmented images
                noisy_images = add_noise(img)
                for idx, noisy_img in enumerate(noisy_images):
                    # Save augmented image
                    aug_img_path = os.path.join(DATASET_ROOT, split, 'images', 
                                              f"{base_name}_aug{idx}{ext}")
                    cv2.imwrite(aug_img_path, noisy_img)
                    saved_count += 1

                    # Copy annotation file for augmented image
                    if os.path.exists(src_ann):
                        aug_ann_path = os.path.join(DATASET_ROOT, split, 'labels',
                                                  f"{base_name}_aug{idx}.txt")
                        shutil.copy2(src_ann, aug_ann_path)

            return saved_count

        # Copy files to respective splits with augmentation
        train_count = copy_files_with_augmentation(train_files, 'train')
        valid_count = copy_files_with_augmentation(valid_files, 'valid')
        test_count = copy_files_with_augmentation(test_files, 'test')

        return jsonify({
            'success': True,
            'message': f'Dataset created with {train_count} training, {valid_count} validation, and {test_count} test images (including augmentations)'
        })

    except Exception as e:
        print(f"Error preparing training data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_annotations', methods=['POST'])
def clear_annotations():
    try:
        data = request.json
        image_file = data['image_file']
        
        # Get base filename without extension
        base_filename = os.path.splitext(image_file)[0]
        
        # Clear annotation file
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, f"{base_filename}.txt")
        if os.path.exists(annotation_path):
            # Write empty file
            open(annotation_path, 'w').close()
        
        return jsonify({'success': True, 'message': 'Annotations cleared successfully'})
    
    except Exception as e:
        print(f"Error clearing annotations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_all_annotations', methods=['POST'])
def clear_all_annotations():
    try:
        # Get absolute path to annotations folder
        annotations_path = os.path.abspath(ANNOTATIONS_FOLDER)
        print(f"Clearing annotations from: {annotations_path}")  # Debug print
        
        # Get all files in the annotations directory
        annotation_files = [f for f in os.listdir(annotations_path) 
                          if f.endswith('.txt')]
        
        deleted_count = 0
        for ann_file in annotation_files:
            file_path = os.path.join(annotations_path, ann_file)
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)  # Using unlink instead of remove
                    deleted_count += 1
                    print(f"Deleted: {file_path}")  # Debug print
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
        
        # Also delete labels.txt if it exists
        labels_path = os.path.abspath(LABELS_FILE)
        if os.path.exists(labels_path):
            os.unlink(labels_path)
            print(f"Deleted labels file: {labels_path}")  # Debug print
        
        print(f"Successfully deleted {deleted_count} annotation files")  # Debug print
        
        return jsonify({
            'success': True,
            'message': f'Successfully deleted {deleted_count} annotation files'
        })
    
    except Exception as e:
        error_msg = f"Error clearing annotations: {str(e)}"
        print(error_msg)  # Debug print
        return jsonify({
            'error': error_msg,
            'success': False
        }), 500

@app.route('/get_video_details', methods=['POST'])
def get_video_details():
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
            
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return jsonify({
            'success': True,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_images', methods=['POST'])
def clear_images():
    try:
        # Clear images from upload folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        return jsonify({
            'success': True,
            'message': 'All images cleared successfully'
        })
    except Exception as e:
        print(f"Error clearing images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/start_process', methods=['POST'])
def start_process():
    try:
        # Initialize training process
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add these constants at the top of your file with other configurations
MODEL_DIR = 'runs/detect/train'
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route('/training_progress')
def training_progress():
    try:
        start_time = time()  # Start timing
        
        dataset_root = os.path.abspath(DATASET_ROOT)
        yaml_path = os.path.join(dataset_root, 'data.yaml')
        
        model = YOLO("yolo11m.pt")
        results = model.train(
            data=yaml_path,
            epochs=100,
            project='runs/detect',
            name='train',
            exist_ok=True
        )
        
        training_time = time() - start_time  # Calculate training time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        model_path = os.path.join(MODEL_DIR, 'weights/last.pt')

        print(f"Training completed in {hours}h {minutes}m {seconds}s")
        
        if os.path.exists(model_path):
            print(f"Model saved at: {model_path}")
            return jsonify({
                'success': True, 
                'message': f'Training completed in {hours}h {minutes}m {seconds}s', 
                'model_path': model_path,
                'training_time': {
                    'hours': hours,
                    'minutes': minutes,
                    'seconds': seconds,
                    'total_seconds': training_time
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Model file not found'}), 500
        
    except Exception as e:
        print(f"Error in training_progress: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_class', methods=['POST'])
def save_class():
    try:
        print("Received save_class request")  # Debug print
        data = request.get_json()
        print("Received data:", data)  # Debug print
        
        class_name = data.get('class_name')
        print(f"Class name received: {class_name}")  # Debug print
        
        if not class_name:
            return jsonify({'error': 'No class name provided'}), 400

        # Read existing classes
        existing_classes = set()
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                existing_classes = set(line.strip() for line in f)

        # Add new class if it doesn't exist
        if class_name not in existing_classes:
            with open(LABELS_FILE, 'a') as f:
                f.write(f"{class_name}\n")
            print(f"Added new class: {class_name}")  # Debug print

        return jsonify({'success': True, 'message': f'Class {class_name} saved successfully'})
    except Exception as e:
        print(f"Error in save_class: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

# Add these global variables at the top with other configurations
TEST_FOLDER = 'static/test'
os.makedirs(TEST_FOLDER, exist_ok=True)

@app.route('/upload_test_files', methods=['POST'])
def upload_test_files():
    print("Received test file upload request")  # Debug print
    
    try:
        if 'files[]' not in request.files:
            print("No files in request")  # Debug print
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files[]')
        uploaded_files = []
        
        # Create test folder if it doesn't exist
        os.makedirs(TEST_FOLDER, exist_ok=True)
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(TEST_FOLDER, filename)
                file.save(filepath)
                uploaded_files.append(filename)
                print(f"Saved test file: {filepath}")  # Debug print
        
        response = {
            'success': True,
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'files': uploaded_files
        }
        print("Upload response:", response)  # Debug print
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in upload_test_files: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

def draw_validation_image(original_img, annotated_img, predictions, confidence_threshold=0.6):
    """Create a side-by-side comparison image with detailed information"""
    
    # Convert BGR to RGB if needed
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Create a larger canvas for side-by-side comparison
    h, w = original_img.shape[:2]
    validation_img = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
    
    # Place original and annotated images
    validation_img[:, :w] = original_img
    validation_img[:, w+20:] = annotated_img
    
    # Add dividing line
    cv2.line(validation_img, (w+10, 0), (w+10, h), (255, 255, 255), 2)
    
    # Add labels
    cv2.putText(validation_img, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(validation_img, "Annotated", (w+30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add detection information
    info_y = 60
    for pred in predictions:
        if pred['confidence'] >= confidence_threshold:
            info = f"Class: {pred['class']}, Conf: {pred['confidence']:.2f}"
            cv2.putText(validation_img, info, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += 20
    
    return validation_img

@app.route('/start_testing', methods=['POST'])
def start_testing():
    try:
        print("Starting testing...")
        start_time = time()
        
        # Create a timestamp-based folder for this test session
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_folder = os.path.join(RESULTS_FOLDER, f'test_session_{timestamp}')
        os.makedirs(session_folder, exist_ok=True)
        
        # Create subfolders for annotated media
        annotated_folder = os.path.join(session_folder, 'annotated_media')
        os.makedirs(annotated_folder, exist_ok=True)
        
        # Get the model path
        model_path = os.path.join(MODEL_DIR, 'weights/last.pt')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404

        # Load the model
        model = YOLO(model_path)
        
        # Get all test files
        test_files = [f for f in os.listdir(TEST_FOLDER) if allowed_file(f)]
        results = []
        
        # Process each test file
        for file in test_files:
            file_path = os.path.join(TEST_FOLDER, file)
            
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                # Process video
                cap = cv2.VideoCapture(file_path)
                
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Create video writer
                output_path = os.path.join(annotated_folder, f'annotated_{file}')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Run inference
                    prediction = model(frame)
                    
                    # Process predictions for this frame
                    for pred in prediction:
                        boxes = pred.boxes
                        for box in boxes:
                            confidence = float(box.conf)
                            if confidence >= CONFIDENCE_THRESHOLD:
                                # Get coordinates and crop
                                bbox = box.xyxy.tolist()[0]
                                x1, y1, x2, y2 = map(int, bbox)
                                crop_img = frame[y1:y2, x1:x2]
                                
                                # Run OCR on crop
                                ocr_text = run_ocr_on_crop(crop_img)
                                
                                # Add to results
                                results.append({
                                    'filename': file,
                                    'frame': frame_count,
                                    'class': model.names[int(box.cls)],
                                    'confidence': confidence,
                                    'bbox': bbox,
                                    'ocr_text': ocr_text
                                })
                                
                                # Draw on frame
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{model.names[int(box.cls)]}: {ocr_text}", 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Write the annotated frame
                    out.write(frame)
                    frame_count += 1
                
                cap.release()
                out.release()
                
            else:
                # Process image
                img = cv2.imread(file_path)
                prediction = model(img)
                annotated_img = img.copy()
                
                # Process predictions
                for pred in prediction:
                    boxes = pred.boxes
                    for box in boxes:
                        confidence = float(box.conf)
                        if confidence >= CONFIDENCE_THRESHOLD:
                            # Get coordinates and crop
                            bbox = box.xyxy.tolist()[0]
                            x1, y1, x2, y2 = map(int, bbox)
                            crop_img = img[y1:y2, x1:x2]
                            
                            # Run OCR on crop
                            ocr_text = run_ocr_on_crop(crop_img)
                            
                            # Add to results
                            results.append({
                                'filename': file,
                                'class': model.names[int(box.cls)],
                                'confidence': confidence,
                                'bbox': bbox,
                                'ocr_text': ocr_text
                            })
                            
                            # Draw on image
                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_img, f"{model.names[int(box.cls)]}: {ocr_text}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save annotated image
                output_path = os.path.join(annotated_folder, f'annotated_{file}')
                cv2.imwrite(output_path, annotated_img)
        
        # Create Excel file with results
        if results:
            df = pd.DataFrame(results)
            excel_path = os.path.join(session_folder, f'test_results_{timestamp}.xlsx')
            
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                # Detailed results
                df.to_excel(writer, sheet_name='Detailed Results', index=False)
                
                # Summary sheet
                summary_data = []
                for file in df['filename'].unique():
                    file_data = df[df['filename'] == file]
                    summary = {
                        'File': file,
                        'Total Detections': len(file_data),
                        'Classes Detected': ', '.join(file_data['class'].unique()),
                        'Average Confidence': file_data['confidence'].mean()
                    }
                    summary_data.append(summary)
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Calculate testing time
        testing_time = time() - start_time
        hours = int(testing_time // 3600)
        minutes = int((testing_time % 3600) // 60)
        seconds = int(testing_time % 60)
        
        # Add timing information to Excel
        timing_df = pd.DataFrame([{
            'Testing Start Time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'Testing Duration': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            'Total Time (seconds)': testing_time
        }])
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl') as writer:
            timing_df.to_excel(writer, sheet_name='Timing Information', index=False)
        
        # Store session info in app config
        app.config['LATEST_SESSION'] = {
            'timestamp': timestamp,
            'folder': session_folder,
            'testing_time': {
                'hours': hours,
                'minutes': minutes,
                'seconds': seconds,
                'total_seconds': testing_time
            }
        }
        
        print(f"Testing completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
        return jsonify({
            'success': True,
            'message': f'Testing completed in {hours:02d}:{minutes:02d}:{seconds:02d}',
            'session_id': timestamp,
            'testing_time': {
                'hours': hours,
                'minutes': minutes,
                'seconds': seconds,
                'total_seconds': testing_time
            }
        })
        
    except Exception as e:
        print(f"Error in testing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_results')
def download_results():
    try:
        if 'LATEST_SESSION' not in app.config:
            return jsonify({'error': 'No test results available'}), 404
        
        session_info = app.config['LATEST_SESSION']
        session_folder = session_info['folder']
        timestamp = session_info['timestamp']
        
        # Create ZIP file
        zip_path = os.path.join(RESULTS_FOLDER, f'test_results_{timestamp}.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Walk through the session folder and add all files
            for folder_path, _, filenames in os.walk(session_folder):
                for filename in filenames:
                    file_path = os.path.join(folder_path, filename)
                    arcname = os.path.relpath(file_path, session_folder)
                    zipf.write(file_path, arcname)
        
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'test_results_{timestamp}.zip'
        )
        
    except Exception as e:
        print(f"Error creating ZIP file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_results/<filename>')
def download_file(filename):
    return send_from_directory(os.path.join(app.static_folder, 'output'), filename)

@app.route('/download_all_results')
def download_all_results():
    try:
        # Create necessary directories if they don't exist
        output_dir = os.path.join(app.static_folder, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Get the latest session info
        if 'LATEST_SESSION' not in app.config:
            return jsonify({'error': 'No test results available'}), 404
            
        session_info = app.config['LATEST_SESSION']
        session_folder = session_info['folder']

        # Create a temporary directory for organizing files
        temp_dir = os.path.join(output_dir, 'temp_zip')
        os.makedirs(temp_dir, exist_ok=True)

        # Create subdirectories in temp folder
        excel_dir = os.path.join(temp_dir, 'excel_results')
        media_dir = os.path.join(temp_dir, 'annotated_media')
        os.makedirs(excel_dir, exist_ok=True)
        os.makedirs(media_dir, exist_ok=True)

        # Copy all files from the session folder to temp directory
        for root, _, files in os.walk(session_folder):
            for file in files:
                src_path = os.path.join(root, file)
                # Determine destination based on file type
                if file.endswith('.xlsx'):
                    dst_path = os.path.join(excel_dir, file)
                else:  # Media files (images and videos)
                    dst_path = os.path.join(media_dir, file)
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)

        # Create the ZIP file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_path = os.path.join(output_dir, f'all_results_{timestamp}.zip')

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add files from temp directory to ZIP
            for folder_path, _, filenames in os.walk(temp_dir):
                for filename in filenames:
                    file_path = os.path.join(folder_path, filename)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        # Send the ZIP file
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'all_results_{timestamp}.zip'
        )

    except Exception as e:
        print(f"Error creating ZIP file: {str(e)}")
        return jsonify({'error': str(e)}), 500

async def async_read_image(path):
    """Async function to read an image"""
    async with aiofiles.open(path, mode='rb') as f:
        img_bytes = await f.read()
        arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def run_ocr_on_crop(crop_img):
    """OCR on cropped image"""
    if crop_img is None or crop_img.size == 0:
        return ""
    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray_img).strip()

async def process_detection_with_ocr(img, prediction, file_path, model):
    """Process YOLO detection and run OCR"""
    results = []
    annotated_img = img.copy()
    
    for pred in prediction:
        boxes = pred.boxes
        for box in boxes:
            conf = float(box.conf)
            if conf >= CONFIDENCE_THRESHOLD:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                bbox = box.xyxy.tolist()[0]
                
                # Crop and OCR
                x1, y1, x2, y2 = map(int, bbox)
                crop_img = img[y1:y2, x1:x2]
                ocr_text = run_ocr_on_crop(crop_img)
                
                # Add to results
                results.append({
                    'filename': os.path.basename(file_path),
                    'class': class_name,
                    'confidence': conf,
                    'bbox': bbox,
                    'text': ocr_text
                })
                
                # Draw on image
                cv2.rectangle(annotated_img, 
                            (x1, y1), (x2, y2),
                            (0, 255, 0), 2)
                label = f'{class_name}: {ocr_text}'
                cv2.putText(annotated_img, 
                          label,
                          (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (0, 255, 0), 2)
    
    return results, annotated_img

@app.route('/delete_project', methods=['POST'])
def delete_project():
    try:
        # Define test directory
        TEST_FOLDER = 'static/test'
        os.makedirs(TEST_FOLDER, exist_ok=True)  # Ensure test folder exists
        
        # List of directories to clean
        directories_to_clean = [
            'runs',
            'dataset',
            'static/uploads',
            'static/videos',
            'static/output',
            'static/verification',
            'annotations',
            'static/test',  # Test folder where images/videos are stored
            'static/test/crops',  # Cropped images from test folder
            'static/test/annotated',  # Annotated results
            'static/test/results'  # Test results including Excel files
        ]
        
        # List of files to delete
        files_to_delete = [
            'labels.txt',
            'data.yaml',
            'detection_results.xlsx'
        ]
        
        # Clean directories
        for directory in directories_to_clean:
            dir_path = os.path.abspath(directory)
            if os.path.exists(dir_path):
                print(f"Deleting directory: {dir_path}")
                try:
                    shutil.rmtree(dir_path)
                    os.makedirs(dir_path, exist_ok=True)  # Recreate empty directory
                    print(f"Successfully cleaned {dir_path}")
                except Exception as e:
                    print(f"Error cleaning {dir_path}: {str(e)}")
        
        # Delete individual files
        for file in files_to_delete:
            file_path = os.path.abspath(file)
            if os.path.exists(file_path):
                print(f"Deleting file: {file_path}")
                try:
                    os.remove(file_path)
                    print(f"Successfully deleted {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")
        
        print("Project deletion completed")
        return jsonify({
            'success': True,
            'message': 'Project data deleted successfully'
        })
        
    except Exception as e:
        error_msg = f"Error deleting project data: {str(e)}"
        print(f"Error in delete_project: {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/retraining_progress')
def retraining_progress():
    try:
        start_time = time()  # Start timing
        
        dataset_root = os.path.abspath(DATASET_ROOT)
        yaml_path = os.path.join(dataset_root, 'data.yaml')
        
        # Load the existing model for retraining
        existing_model_path = os.path.join(MODEL_DIR, 'weights/last.pt')
        if not os.path.exists(existing_model_path):
            return jsonify({'error': 'No existing model found for retraining'}), 404
        
        model = YOLO(existing_model_path)
        results = model.train(
            data=yaml_path,
            epochs=100,
            project='runs/detect',
            name='retrain',
            exist_ok=True
        )
        
        training_time = time() - start_time  # Calculate training time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        # Save the retrained model in a different directory
        retrain_dir = 'runs/detect/retrain'
        model_path = os.path.join(retrain_dir, 'weights/last.pt')

        print(f"Retraining completed in {hours}h {minutes}m {seconds}s")
        
        if os.path.exists(model_path):
            print(f"Retrained model saved at: {model_path}")
            return jsonify({
                'success': True, 
                'message': f'Retraining completed in {hours}h {minutes}m {seconds}s', 
                'model_path': model_path,
                'training_time': {
                    'hours': hours,
                    'minutes': minutes,
                    'seconds': seconds,
                    'total_seconds': training_time
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Model file not found'}), 500
        
    except Exception as e:
        print(f"Error in retraining_progress: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Add with other configuration constants
RESULTS_FOLDER = 'static/results'
os.makedirs(RESULTS_FOLDER, exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Disable reloader