# pipeline.py
import os
import json
from model import DefectDetectionModel
from config import DEFECT_PROMPTS
from helper import get_directory_path, get_image_files


class DefectDetectionPipeline:
    def __init__(self):
        self.model = DefectDetectionModel()
    
    def process_defect_type(self, defect_name, output_file=None):
        """Process all images for a specific defect type"""
        
        # Get prompt and directory path
        prompt = DEFECT_PROMPTS.get(defect_name)
        directory_path = get_directory_path(defect_name)
        
        if not prompt:
            print(f"Error: No prompt found for defect type '{defect_name}'")
            return {}
        
        if not directory_path or not os.path.exists(directory_path):
            print(f"Error: Directory not found for defect type '{defect_name}'")
            return {}
        
        # Get image files
        image_files = get_image_files(directory_path)
        
        if not image_files:
            print(f"No image files found in directory: {directory_path}")
            return {}
        
        results = {}
        print(f"Processing {len(image_files)} images from {directory_path}")
        
        # Process each image
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(directory_path, filename)
            print(f"Processing {i}/{len(image_files)}: {filename}")
            
            try:
                result = self.model.predict(image_path, prompt)
                results[filename] = {
                    "status": "success",
                    "result": result
                }
                print(f"✓ Result: {result}")
                
            except Exception as e:
                results[filename] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"✗ Error: {str(e)}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results

def main():
    """Main function - Change defect_name here to process different defect types"""
    
    # ============================================
    # CHANGE THIS VARIABLE TO PROCESS DIFFERENT DEFECT TYPES
    # Available options: 'ceiling_panel', 'frosted_window', 'missing_grab_handle', 
    #                   'missing_lighting_panel', 'switch_cover'
    # ============================================
    defect_name = "missing_lighting_panel"  # <-- Change this to process different defects
    
    # Initialize pipeline
    pipeline = DefectDetectionPipeline()
    
    # Process the specified defect type
    results = pipeline.process_defect_type(
        defect_name=defect_name,
        output_file=f"{defect_name}_analysis_results.json"
    )
    
    # Print summary
    if results:
        successful = sum(1 for r in results.values() if r["status"] == "success")
        print(f"Completed: {successful}/{len(results)} images processed successfully")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()