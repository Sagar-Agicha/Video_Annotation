import cv2
import os
import json
import numpy as np
from pathlib import Path

class AnnotationVerifier:
    def __init__(self, image_dir, annotation_dir, output_dir):
        """
        Initialize the verifier
        
        Args:
            image_dir (str): Directory containing original images
            annotation_dir (str): Directory containing annotation files
            output_dir (str): Directory to save annotated images
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Colors for different classes (will be assigned dynamically)
        self.colors = {}
        
    def generate_color(self, class_name):
        """Generate a unique color for each class"""
        if class_name not in self.colors:
            # Generate random color
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            self.colors[class_name] = color
        return self.colors[class_name]
    
    def yolo_to_pixel_coords(self, x_center, y_center, width, height, img_width, img_height):
        """Convert YOLO coordinates to pixel coordinates"""
        # YOLO format uses center coordinates and normalized dimensions
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width)
        height = float(height)
        
        # Convert to pixel values
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # Calculate top-left corner
        x1 = int(x_center_px - (width_px / 2))
        y1 = int(y_center_px - (height_px / 2))
        
        # Calculate bottom-right corner
        x2 = int(x_center_px + (width_px / 2))
        y2 = int(y_center_px + (height_px / 2))
        
        return x1, y1, x2, y2
    
    def draw_annotation(self, image, bbox, class_name, confidence=None):
        """Draw a single annotation on the image"""
        x1, y1, x2, y2 = bbox
        color = self.generate_color(class_name)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = class_name
        if confidence is not None:
            label += f" {confidence:.2f}"
            
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1-20), (x1+label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw center point for verification
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)
    
    def create_side_by_side(self, original, annotated):
        """Create a side-by-side comparison image"""
        h, w = original.shape[:2]
        combined = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
        
        # Copy images
        combined[:, :w] = original
        combined[:, w+20:] = annotated
        
        # Add dividing line
        cv2.line(combined, (w+10, 0), (w+10, h), (255, 255, 255), 2)
        
        # Add labels
        cv2.putText(combined, "Original", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Annotated", (w+30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined
    
    def process_image(self, image_path):
        """Process a single image and its annotations"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error loading image: {image_path}")
            return
        
        # Get corresponding annotation file
        ann_file = self.annotation_dir / f"{image_path.stem}.txt"
        if not ann_file.exists():
            print(f"No annotation file found for: {image_path}")
            return
        
        # Create copy for annotation
        annotated = image.copy()
        h, w = image.shape[:2]
        
        # Read and process annotations
        with open(ann_file, 'r') as f:
            for line in f:
                try:
                    # Parse YOLO format
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Convert to pixel coordinates
                        bbox = self.yolo_to_pixel_coords(x_center, y_center, width, height, w, h)
                        
                        # Draw annotation
                        self.draw_annotation(annotated, bbox, f"Class {class_id}")
                        
                except ValueError as e:
                    print(f"Error processing annotation in {ann_file}: {e}")
                    continue
        
        # Create side-by-side comparison
        combined = self.create_side_by_side(image, annotated)
        
        # Save result
        output_path = self.output_dir / f"verified_{image_path.name}"
        cv2.imwrite(str(output_path), combined)
        
        return output_path
    
    def process_all(self):
        """Process all images in the directory"""
        processed_files = []
        image_files = list(self.image_dir.glob('*.jpg')) + \
                     list(self.image_dir.glob('*.jpeg')) + \
                     list(self.image_dir.glob('*.png'))
        
        total_files = len(image_files)
        print(f"Found {total_files} images to process")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\nProcessing {i}/{total_files}: {image_path.name}")
            output_path = self.process_image(image_path)
            if output_path:
                processed_files.append(output_path)
        
        return processed_files

def main():
    # Configure directories
    image_dir = "static/uploads"  # Directory with original images
    annotation_dir = "annotations"  # Directory with annotation files
    output_dir = "static/verification"  # Directory for output images
    
    # Create verifier instance
    verifier = AnnotationVerifier(image_dir, annotation_dir, output_dir)
    
    # Process all images
    processed_files = verifier.process_all()
    
    # Print summary
    print("\nVerification Complete!")
    print(f"Processed {len(processed_files)} files")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main() 