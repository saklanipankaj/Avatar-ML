# helper.py
import os

# Directory paths for each defect type
DEFECT_DIRECTORIES = {
    "ceiling_panel": "./ceiling_panel",
    "frosted_window": "./frosted_window", 
    "missing_grab_handle": "./missing_grab_handle",
    "missing_lighting_panel": "./missing_lighting_panel",
    "switch_cover": "./switch_cover"
}

def get_directory_path(defect_name):
    """Get directory path for given defect type"""
    return DEFECT_DIRECTORIES.get(defect_name)

def get_image_files(directory_path):
    """Get all image files from directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    if not os.path.exists(directory_path):
        return []
    
    return [
        f for f in os.listdir(directory_path)
        if any(f.endswith(ext) for ext in image_extensions)
    ]

def get_image_media_type(image_path):
    """Determine media type based on file extension"""
    if image_path.lower().endswith(('.png', '.PNG')):
        return "image/png"
    elif image_path.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
        return "image/jpeg"
    else:
        return "image/jpeg"  # default