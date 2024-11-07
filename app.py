from flask import Flask, render_template, request, jsonify, send_from_directory, Response
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


app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
VIDEO_FOLDER = 'static/videos'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'wmv'}
print("ALLOWED_EXTENSIONS: ", ALLOWED_EXTENSIONS)

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
                        class_set.add(class_num - 1)  # Subtract 1 to convert to 0-based index
        
        # Read labels.txt to get class names
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                class_names = {i: name.strip() for i, name in enumerate(f.readlines())}
        
        print(f"Found classes: {class_names}")  # Debug print
        
        # Create data.yaml file
        yaml_content = {
            'path': os.path.abspath(DATASET_ROOT),
            'train': os.path.join('train', 'images'),
            'val': os.path.join('valid', 'images'),
            'test': os.path.join('test', 'images'),
            'nc': len(class_names),
            'names': [class_names[i] for i in range(len(class_names))]
        }

        print(f"YAML content: {yaml_content}")  # Debug print

        yaml_path = os.path.join(DATASET_ROOT, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False)

        # Get all images and their annotations
        image_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
        
        # Split data
        train_files, temp_files = train_test_split(image_files, train_size=0.7, random_state=42)
        valid_files, test_files = train_test_split(temp_files, train_size=0.67, random_state=42)

        def copy_files(files, split):
            for img_file in files:
                # Copy image
                src_img = os.path.join(app.config['UPLOAD_FOLDER'], img_file)
                dst_img = os.path.join(DATASET_ROOT, split, 'images', img_file)
                shutil.copy2(src_img, dst_img)

                # Copy and adjust annotation if it exists
                base_name = os.path.splitext(img_file)[0]
                ann_file = f"{base_name}.txt"
                src_ann = os.path.join(ANNOTATIONS_FOLDER, ann_file)
                if os.path.exists(src_ann):
                    dst_ann = os.path.join(DATASET_ROOT, split, 'labels', ann_file)
                    # Adjust class numbers to start from 0
                    with open(src_ann, 'r') as f_in, open(dst_ann, 'w') as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            class_num = int(parts[0])
                            # Subtract 1 from class number
                            adjusted_line = f"{class_num - 1} {' '.join(parts[1:])}\n"
                            f_out.write(adjusted_line)

        # Copy files to respective splits
        copy_files(train_files, 'train')
        copy_files(valid_files, 'valid')
        copy_files(test_files, 'test')

        return jsonify({
            'success': True,
            'message': f'Dataset created with {len(train_files)} training, {len(valid_files)} validation, and {len(test_files)} test images'
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

@app.route('/training_progress')
def training_progress():
    def generate():
        try:
            # Disable reloader for this process
            if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
                dataset_root = os.path.abspath(DATASET_ROOT)
                yaml_path = os.path.join(dataset_root, 'data.yaml')
                
                model = YOLO("C://Users/DELL/Desktop/Galaxy/VIdeo Annotations/yolo11n.pt")
                results = model.train(data=yaml_path, epochs=100)

                return jsonify({'status': 'completed'})
            
        except Exception as e:
            print(f"Error in training_progress: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

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

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Disable reloader