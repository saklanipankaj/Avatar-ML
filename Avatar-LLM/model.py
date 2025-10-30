# model.py
import boto3
import json
import base64
from botocore.config import Config
from helper import get_image_media_type

class DefectDetectionModel:
    def __init__(self):
        # Configuration
        #self.inference_profile_arn = "arn:aws:bedrock:ap-southeast-1:946156973544:inference-profile/apac.anthropic.claude-3-7-sonnet-20250219-v1:0"
        #self.inference_profile_arn = "arn:aws:bedrock:ap-southeast-1:946156973544:inference-profile/apac.anthropic.claude-sonnet-4-20250514-v1:0"
        self.inference_profile_arn = "arn:aws:bedrock:ap-southeast-1:946156973544:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        # Define retry configuration
        retry_config = Config(
            retries={
                'max_attempts': 5,
                'mode': 'standard'
            },
            connect_timeout=120
        )
        
        # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name='ap-southeast-1',
            config=retry_config
        )
    
    def predict(self, image_path, prompt):
        """Process single image with given prompt"""
        try:
            # Read and encode image
            with open(image_path, "rb") as img:
                base64_image = base64.b64encode(img.read()).decode("utf-8")
            
            media_type = get_image_media_type(image_path)
            
            # Construct messages
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
            
            # Payload for Claude
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "temperature": 0,
                "messages": messages
            }
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                modelId=self.inference_profile_arn,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload)
            )
            
            # Parse response
            result = json.loads(response['body'].read().decode('utf-8'))
            return result['content'][0]['text']
            
        except Exception as e:
            raise Exception(f"Model prediction failed: {str(e)}")

    def predict_fewshot(self, image_paths, prompt, max_tokens=512, temperature=0):
        """Process multiple images as a few-shot example set + a query image.

        image_paths: list of tuples (label, path) or list of paths. If labels
        are provided they are embedded as text examples along with images.
        The last image in the list will be treated as the query unless the
        caller includes an explicit marker in the prompt.
        """
        try:
            # Build content list with image objects and optional labels
            image_objects = []

            for item in image_paths:
                # item may be (label, path) or just path
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    label, path = item
                else:
                    label, path = None, item

                with open(path, "rb") as img:
                    b64 = base64.b64encode(img.read()).decode("utf-8")

                media_type = get_image_media_type(path)

                obj = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64
                    }
                }

                # If a label is provided, include it as a text object after the image
                image_objects.append(obj)
                if label:
                    image_objects.append({"type": "text", "text": f"Label: {label}"})

            # Append the query prompt text at the end
            image_objects.append({"type": "text", "text": prompt})

            messages = [{
                "role": "user",
                "content": image_objects
            }]

            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.inference_profile_arn,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload)
            )

            result = json.loads(response['body'].read().decode('utf-8'))
            return result['content'][0]['text']

        except Exception as e:
            raise Exception(f"Few-shot model prediction failed: {str(e)}")