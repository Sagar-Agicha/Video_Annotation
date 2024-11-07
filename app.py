from flask import Flask, render_template, request, jsonify, send_from_directory
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

# Add these constants at the top of your file
DATASET_ROOT = 'dataset'
IMAGES_DIR = os.path.join(DATASET_ROOT, 'images')
LABELS_DIR = os.path.join(DATASET_ROOT, 'labels')

# Create dataset directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(IMAGES_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

@app.route('/prepare_training', methods=['POST'])
def prepare_training():
    try:
        # Create dataset directories if they don't exist
        os.makedirs(DATASET_ROOT, exist_ok=True)
        os.makedirs(IMAGES_DIR, exist_ok=True)
        os.makedirs(LABELS_DIR, exist_ok=True)

        # Get all image files
        image_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Prepare file pairs (image and annotation)
        file_pairs = []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            ann_file = f"{base_name}.txt"
            
            if os.path.exists(os.path.join('annotations', ann_file)):
                file_pairs.append((img_file, ann_file))

        # Shuffle the pairs
        random.shuffle(file_pairs)

        # Calculate split sizes
        total = len(file_pairs)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)
        test_size = total - train_size - val_size

        # Split the data
        train_pairs = file_pairs[:train_size]
        val_pairs = file_pairs[train_size:train_size + val_size]
        test_pairs = file_pairs[train_size + val_size:]

        # Function to move files to their respective directories
        def move_pairs(pairs, split_name):
            for img_file, ann_file in pairs:
                # Move image
                src_img = os.path.join(app.config['UPLOAD_FOLDER'], img_file)
                dst_img = os.path.join(IMAGES_DIR, split_name, img_file)
                shutil.copy2(src_img, dst_img)

                # Move annotation
                src_ann = os.path.join('annotations', ann_file)
                dst_ann = os.path.join(LABELS_DIR, split_name, ann_file)
                shutil.copy2(src_ann, dst_ann)

        # Move files to their respective directories
        move_pairs(train_pairs, 'train')
        move_pairs(val_pairs, 'val')
        move_pairs(test_pairs, 'test')

        # Create dataset.yaml file
        yaml_content = f"""
path: {os.path.abspath(DATASET_ROOT)}
train: images/train
val: images/val
test: images/test

names:
{create_yaml_names()}
        """

        with open(os.path.join(DATASET_ROOT, 'dataset.yaml'), 'w') as f:
            f.write(yaml_content)

        return jsonify({
            'success': True,
            'message': f'Dataset prepared successfully with {len(train_pairs)} training, {len(val_pairs)} validation, and {len(test_pairs)} test samples'
        })

    except Exception as e:
        print(f"Error preparing training data: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_yaml_names():
    """Create the names section of the YAML file from labels"""
    yaml_names = []
    with open(LABELS_FILE, 'r') as f:
        for idx, label in enumerate(f):
            yaml_names.append(f"  {idx}: {label.strip()}")
    return '\n'.join(yaml_names)

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
        # Get all annotation files
        annotation_files = [f for f in os.listdir(ANNOTATIONS_FOLDER) 
                          if f.endswith('.txt')]
        
        # Clear each annotation file
        for ann_file in annotation_files:
            file_path = os.path.join(ANNOTATIONS_FOLDER, ann_file)
            try:
                # Option 1: Clear the file content
                open(file_path, 'w').close()
                
                # Option 2: Delete the file (uncomment if you prefer deletion)
                # os.remove(file_path)
            except Exception as e:
                print(f"Error clearing annotation file {ann_file}: {str(e)}")
        
        # Also clear the labels.txt file if you want to reset class labels
        if os.path.exists(LABELS_FILE):
            open(LABELS_FILE, 'w').close()
        
        return jsonify({
            'success': True,
            'message': f'Cleared {len(annotation_files)} annotation files'
        })
    
    except Exception as e:
        print(f"Error clearing all annotations: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Cleared all annotations'
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
        # Get absolute paths
        dataset_root = os.path.abspath(DATASET_ROOT)
        train_path = os.path.abspath(os.path.join(IMAGES_DIR, 'train'))
        val_path = os.path.abspath(os.path.join(IMAGES_DIR, 'val'))
        test_path = os.path.abspath(os.path.join(IMAGES_DIR, 'test'))
        yaml_path = os.path.abspath(os.path.join(DATASET_ROOT, 'dataset.yaml'))

        model = YOLO("yolo11l.pt")
        results = model.train(data=yaml_path,epochs=100)
        
        return jsonify({
            'success': True,
            'paths': {
                'dataset_root': dataset_root,
                'train': train_path,
                'val': val_path,
                'test': test_path,
                'yaml': yaml_path
            }
        })
        
    except Exception as e:
        print(f"Error in start_process: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)