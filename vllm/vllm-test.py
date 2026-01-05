#!/usr/bin/env python3
"""
Test script for vLLM image analysis using OpenAI-compatible API
Demonstrates sending an image for analysis to the vLLM server
"""

import base64
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed")
    print("Install with: pip install openai")
    sys.exit(1)


def encode_image(image_path: str) -> str:
    """Read and encode image as base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_image(image_path: str, prompt: str = "What's in this image? Describe it in detail.", 
                  server_url: str = "http://localhost:8000/v1"):
    """
    Send an image to vLLM for analysis
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt for the analysis
        server_url: vLLM server URL (format: http://host:port/v1)
    """
    if not Path(image_path).exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)
    
    print(f"Analyzing image: {image_path}")
    print(f"Server: {server_url}")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    # Initialize OpenAI client pointing to vLLM
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require an API key
        base_url=server_url
    )
    
    # Encode the image
    print("Encoding image...")
    image_base64 = encode_image(image_path)
    
    # Send request to vLLM
    print("Sending request to vLLM server...")
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2-VL-7B-Instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }],
            max_tokens=512,
            temperature=0.7
        )
        
        # Print the response
        print("\n" + "=" * 60)
        print("ANALYSIS RESULT:")
        print("=" * 60)
        print(response.choices[0].message.content)
        print("=" * 60)
        print(f"\nTokens used: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"\nERROR: Failed to analyze image")
        print(f"Error: {e}")
        print("\nMake sure the vLLM server is running. Check with:")
        print("  docker logs vllm-qwen2vl")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test-image-analysis.py <image_path> [prompt] [server_url]")
        print("\nExamples:")
        print("  python test-image-analysis.py photo.jpg")
        print("  python test-image-analysis.py document.png 'Extract all text from this document'")
        print("  python test-image-analysis.py chart.jpg 'Analyze this chart' http://192.168.1.100:8000/v1")
        sys.exit(1)
    
    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "What's in this image? Describe it in detail."
    server_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000/v1"
    
    analyze_image(image_path, prompt, server_url)
