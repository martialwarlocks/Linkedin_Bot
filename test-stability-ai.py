#!/usr/bin/env python3
"""
Test script for Stability AI integration
"""

import requests
import json
import os

# Configuration
STABILITY_API_KEY = "sk-Fu8BOqrEBB10vrRBc2sRTtZeMSBbl9NJKCoFzmqHbIvdcIma"
BACKEND_URL = "http://localhost:8080"  # Update this to your backend URL

def test_stability_ai_direct():
    """Test Stability AI API directly"""
    print("🧪 Testing Stability AI API directly...")
    
    api_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "text_prompts": [
            {
                "text": "Professional bar chart showing quarterly revenue growth, clean design with clear labels, business presentation style",
                "weight": 1.0
            }
        ],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30,
        "style_preset": "professional"
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        print("✅ Stability AI API test successful!")
        print(f"   Response status: {response.status_code}")
        print(f"   Artifacts generated: {len(result.get('artifacts', []))}")
        
        if result.get('artifacts'):
            artifact = result['artifacts'][0]
            print(f"   Image size: {len(artifact.get('base64', ''))} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Stability AI API test failed: {e}")
        return False

def test_backend_image_generation():
    """Test the backend image generation endpoint"""
    print("\n🧪 Testing backend image generation endpoint...")
    
    endpoint = f"{BACKEND_URL}/linkedin/generate-image"
    
    payload = {
        "prompt": "Professional bar chart showing quarterly revenue growth, clean design with clear labels, business presentation style",
        "style": "professional",
        "aspect_ratio": "16:9",
        "size": "1024x1024",
        "content_type": "chart"
    }
    
    try:
        response = requests.post(endpoint, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Backend image generation test successful!")
            print(f"   Image URL: {result.get('image_url', 'N/A')}")
            print(f"   Message: {result.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Backend test failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

def test_backend_images_list():
    """Test the backend images listing endpoint"""
    print("\n🧪 Testing backend images listing endpoint...")
    
    endpoint = f"{BACKEND_URL}/linkedin/images"
    
    try:
        response = requests.get(endpoint, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Backend images listing test successful!")
            print(f"   Images count: {result.get('count', 0)}")
            return True
        else:
            print(f"❌ Backend images listing test failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Backend images listing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Stability AI Integration Tests\n")
    
    # Test 1: Direct Stability AI API
    stability_test = test_stability_ai_direct()
    
    # Test 2: Backend image generation
    backend_gen_test = test_backend_image_generation()
    
    # Test 3: Backend images listing
    backend_list_test = test_backend_images_list()
    
    # Summary
    print("\n📊 Test Results Summary:")
    print(f"   Stability AI API: {'✅ PASS' if stability_test else '❌ FAIL'}")
    print(f"   Backend Generation: {'✅ PASS' if backend_gen_test else '❌ FAIL'}")
    print(f"   Backend Listing: {'✅ PASS' if backend_list_test else '❌ FAIL'}")
    
    if all([stability_test, backend_gen_test, backend_list_test]):
        print("\n🎉 All tests passed! Stability AI integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration and try again.")

if __name__ == "__main__":
    main() 