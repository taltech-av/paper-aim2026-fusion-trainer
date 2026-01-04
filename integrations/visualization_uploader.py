import requests
import json
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv
import urllib3

# Disable SSL warnings for HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

VISION_API_BASE_URL = "https://vision-api.visin.eu/api"

def get_auth_headers():
    """Get authorization headers for API requests."""
    token = os.getenv('VISIN_TOKEN')
    if token:
        return {'Authorization': f'Bearer {token}'}
    return {}

def upload_visualization(epoch_uuid, file_path, viz_type, metadata=None):
    """
    Upload a visualization file to the vision service.
    
    This function:
    1. Requests a signed upload URL from the API
    2. Uploads the file directly to MinIO
    3. Creates the visualization record in the database
    
    Args:
        epoch_uuid (str): UUID of the epoch this visualization belongs to
        file_path (str): Path to the visualization file
        viz_type (str): Type of visualization (e.g., 'segment', 'overlay', 'compare', 'correct_only')
        metadata (dict, optional): Additional metadata for the visualization
    
    Returns:
        dict or None: Visualization data if successful, None if failed
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    # Determine mimetype based on extension
    ext = Path(file_path).suffix.lower()
    mimetype_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg'
    }
    mimetype = mimetype_map.get(ext, 'image/png')
    
    try:
        # Step 1: Get signed upload URL
        viz_uuid = str(uuid.uuid4())
        upload_url_request = {
            "epoch_uuid": epoch_uuid,
            "filename": filename,
            "type": viz_type,
            "mimetype": mimetype
        }
        
        url = f"{VISION_API_BASE_URL}/visualizations/upload-url"
        response = requests.post(url, json=upload_url_request, headers=get_auth_headers(), timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success"):
            print(f"Failed to get upload URL: {data.get('message')}")
            return None
        
        upload_url = data["data"]["uploadUrl"]
        viz_uuid = data["data"]["visualization_uuid"]
        minio_file_id = data["data"]["minioFileId"]
        
        # Step 2: Upload file to MinIO
        with open(file_path, 'rb') as f:
            upload_response = requests.put(
                upload_url, 
                data=f,
                headers={'Content-Type': mimetype},
                timeout=30
            )
            upload_response.raise_for_status()
        
        # Step 3: Create visualization record
        create_request = {
            "epoch_uuid": epoch_uuid,
            "visualization_uuid": viz_uuid,
            "filename": filename,
            "type": viz_type,
            "minioFileId": minio_file_id,
            "mimetype": mimetype,
            "size": file_size
        }
        
        if metadata:
            create_request["metadata"] = metadata
        
        url = f"{VISION_API_BASE_URL}/visualizations"
        response = requests.post(url, json=create_request, headers=get_auth_headers(), timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            print(f"Successfully uploaded {viz_type} visualization: {filename}")
            return data.get("data")
        else:
            print(f"Failed to create visualization record: {data.get('message')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed when uploading visualization: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error when uploading visualization: {e}")
        return None


def upload_all_visualizations_for_image(epoch_uuid, output_base, image_name):
    """
    Upload all visualization types for a single image.
    
    Args:
        epoch_uuid (str): UUID of the epoch
        output_base (str): Base directory containing visualization subdirectories
        image_name (str): Name of the image file
    
    Returns:
        dict: Dictionary mapping viz_type to upload result
    """
    viz_types = ['segment', 'overlay', 'compare', 'correct_only']
    results = {}
    
    for viz_type in viz_types:
        file_path = os.path.join(output_base, viz_type, image_name)
        if os.path.exists(file_path):
            result = upload_visualization(
                epoch_uuid=epoch_uuid,
                file_path=file_path,
                viz_type=viz_type,
                metadata={
                    "image_name": image_name,
                    "source": "visualization_script"
                }
            )
            results[viz_type] = result
        else:
            print(f"Skipping {viz_type} - file not found: {file_path}")
            results[viz_type] = None
    
    return results


def get_epoch_uuid_from_model_path(model_path):
    """
    Extract epoch UUID from model checkpoint filename.
    
    Expected format: epoch_{num}_{uuid}.pth
    
    Args:
        model_path (str): Path to the model checkpoint
    
    Returns:
        str or None: Epoch UUID if found, None otherwise
    """
    try:
        filename = os.path.basename(model_path)
        # Remove .pth extension
        name_without_ext = filename.replace('.pth', '')
        # Split by underscore and get the last part (UUID)
        parts = name_without_ext.split('_')
        if len(parts) >= 3 and parts[0] == 'epoch':
            epoch_uuid = '_'.join(parts[2:])  # Handle UUIDs with underscores
            return epoch_uuid
    except Exception as e:
        print(f"Failed to extract epoch UUID from model path: {e}")
    return None
