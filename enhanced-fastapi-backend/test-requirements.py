#!/usr/bin/env python3
"""Test script to verify all requirements can be imported"""

import sys
import traceback

def test_imports():
    """Test importing all required packages"""
    packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'httpx',
        'google.cloud.storage',
        'openai',
        'supabase',
        'fitz',  # PyMuPDF
        'docx',
        'openpyxl',
        'pptx',
        'trafilatura',
        'bs4',  # beautifulsoup4
        'numpy',
        'faiss',
        'tiktoken',
        'sentence_transformers',
        'requests'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
        except Exception as e:
            print(f"⚠️ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {failed_imports}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 