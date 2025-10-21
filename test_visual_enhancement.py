#!/usr/bin/env python3
"""
Test script for the enhanced visual content generation feature
"""

import asyncio
import json
import sys
import os

# Add the current directory to the path so we can import from main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import (
    search_web_for_images,
    generate_contextual_image,
    create_hybrid_visual_content,
    create_data_driven_visualization,
    analyze_research_data
)

async def test_enhanced_visual_generation():
    """Test the enhanced visual content generation system"""
    
    print("🎨 Testing Enhanced Visual Content Generation System")
    print("=" * 60)
    
    # Test 1: Enhanced Image Search
    print("\n1. Testing Enhanced Image Search...")
    test_topics = [
        "AI in business transformation",
        "digital marketing strategies", 
        "leadership development",
        "data analytics trends",
        "startup innovation"
    ]
    
    for topic in test_topics:
        print(f"\n   Topic: {topic}")
        try:
            image_url, source_link = await search_web_for_images(topic)
            if image_url:
                print(f"   ✅ Found image: {image_url[:50]}...")
                print(f"   📎 Source: {source_link}")
            else:
                print("   ❌ No image found")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 2: Contextual Image Generation
    print("\n\n2. Testing Contextual Image Generation...")
    
    # Sample content data
    sample_content = {
        'linkedin_posts': [
            "AI is revolutionizing how businesses operate, from automating routine tasks to providing deep insights through data analysis.",
            "The key to successful digital transformation lies in understanding your customers' needs and leveraging technology to meet them."
        ],
        'video_scripts': [
            "Today we're discussing how AI is changing the business landscape..."
        ],
        'hashtags': ['#AI', '#Business', '#Innovation'],
        'engagement_tips': ['Post during peak hours', 'Engage with comments']
    }
    
    # Sample research data
    sample_research = [
        {
            'id': 1,
            'topic': 'AI Business Impact',
            'findings': 'Companies using AI see 20% increase in productivity and 15% reduction in operational costs.',
            'data': 'Survey of 500 companies across various industries',
            'source': 'Tech Research Institute',
            'tags': 'ai,business,productivity'
        },
        {
            'id': 2,
            'topic': 'Digital Transformation',
            'findings': '78% of businesses report improved customer satisfaction after digital transformation initiatives.',
            'data': 'Customer satisfaction scores increased from 3.2 to 4.1 out of 5',
            'source': 'Digital Business Report 2024',
            'tags': 'digital,transformation,customer'
        }
    ]
    
    try:
        print("   Generating contextual image for sample content...")
        image_url, source_link = await generate_contextual_image(sample_content, sample_research)
        if image_url:
            print(f"   ✅ Generated contextual image: {image_url[:50]}...")
            print(f"   📎 Source: {source_link}")
        else:
            print("   ❌ No contextual image generated")
    except Exception as e:
        print(f"   ❌ Error in contextual generation: {e}")
    
    # Test 3: Hybrid Visual Content
    print("\n\n3. Testing Hybrid Visual Content Generation...")
    
    try:
        print("   Creating hybrid visual content...")
        hybrid_url, hybrid_source = await create_hybrid_visual_content(
            sample_content, 
            sample_research, 
            "AI business transformation"
        )
        if hybrid_url:
            print(f"   ✅ Generated hybrid visual: {hybrid_url[:50]}...")
            print(f"   📎 Source: {hybrid_source}")
        else:
            print("   ❌ No hybrid visual generated")
    except Exception as e:
        print(f"   ❌ Error in hybrid generation: {e}")
    
    # Test 4: Data-Driven Visualization
    print("\n\n4. Testing Data-Driven Visualization...")
    
    try:
        print("   Creating data-driven visualization...")
        data_viz_url = create_data_driven_visualization(sample_research)
        if data_viz_url:
            print(f"   ✅ Generated data visualization: {data_viz_url[:50]}...")
        else:
            print("   ❌ No data visualization generated")
    except Exception as e:
        print(f"   ❌ Error in data visualization: {e}")
    
    # Test 5: Research Data Analysis
    print("\n\n5. Testing Research Data Analysis...")
    
    try:
        print("   Analyzing research data...")
        analysis = analyze_research_data(sample_research)
        print(f"   ✅ Analysis completed:")
        print(f"      - Topics: {analysis['topics']}")
        print(f"      - Word counts: {analysis['word_counts']}")
        print(f"      - Numerical data: {analysis['numerical_data']}")
        print(f"      - Source types: {analysis['source_types']}")
    except Exception as e:
        print(f"   ❌ Error in research analysis: {e}")
    
    print("\n\n🎉 Visual Enhancement Testing Complete!")
    print("=" * 60)
    print("\nKey Features Tested:")
    print("✅ Enhanced image search with topic categorization")
    print("✅ Contextual image generation from content")
    print("✅ Hybrid visual content (images + data overlays)")
    print("✅ Data-driven visualizations from research")
    print("✅ Research data analysis and insights")
    
    print("\n🚀 Your LinkedIn content creation system now includes:")
    print("• Smart image selection based on content topics")
    print("• Professional images from curated databases")
    print("• Data visualizations from your research")
    print("• Hybrid content combining images with insights")
    print("• Multiple fallback strategies for visual content")

if __name__ == "__main__":
    asyncio.run(test_enhanced_visual_generation())



