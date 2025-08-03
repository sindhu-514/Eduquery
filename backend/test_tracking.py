#!/usr/bin/env python3
"""
Test script to demonstrate the content tracking system
This script shows how the system tracks content retrieval success/failure
"""

import requests
import json
import time

# API endpoint
BASE_URL = "http://localhost:8007"

def test_query_with_tracking(query, book_name, session_id="test_session"):
    """
    Test the tracking endpoint and display results
    """
    print(f"\n{'='*60}")
    print(f"TESTING QUERY: '{query}'")
    print(f"BOOK: {book_name}")
    print(f"SESSION: {session_id}")
    print(f"{'='*60}")
    
    try:
        # Make request to tracking endpoint
        response = requests.post(
            f"{BASE_URL}/query_with_tracking/",
            data={
                "query": query,
                "book_name": book_name,
                "session_id": session_id
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Display summary
            summary = data.get("summary", {})
            print(f"\n📊 SUMMARY:")
            print(f"   Content Found: {summary.get('content_found', False)}")
            print(f"   Chunks Retrieved: {summary.get('chunks_retrieved', 0)}")
            print(f"   Response Type: {summary.get('response_type', 'unknown')}")
            print(f"   Success: {summary.get('success', False)}")
            
            # Display answer
            answer = data.get("answer", "No answer")
            print(f"\n🤖 ANSWER:")
            print(f"   {answer[:200]}{'...' if len(answer) > 200 else ''}")
            
            # Display detailed tracking data
            tracking = data.get("tracking_data", {})
            if tracking:
                print(f"\n🔍 DETAILED TRACKING:")
                
                # Content retrieval info
                retrieval = tracking.get("processing_steps", {}).get("content_retrieval", {})
                print(f"   📥 Content Retrieval:")
                print(f"      - Search Strategy: {retrieval.get('search_strategy', 'unknown')}")
                print(f"      - Chunks Retrieved: {retrieval.get('chunks_retrieved', 0)}")
                print(f"      - Content Found: {retrieval.get('content_found', False)}")
                if retrieval.get('error'):
                    print(f"      - Error: {retrieval.get('error')}")
                
                # Response generation info
                response_gen = tracking.get("processing_steps", {}).get("response_generation", {})
                print(f"   🤖 Response Generation:")
                print(f"      - Response Type: {response_gen.get('response_type', 'unknown')}")
                print(f"      - Is Followup: {response_gen.get('is_followup', False)}")
                print(f"      - Is New Topic: {response_gen.get('is_new_topic', False)}")
                if response_gen.get('error'):
                    print(f"      - Error: {response_gen.get('error')}")
                
                # Final result
                final = tracking.get("processing_steps", {}).get("final_result", {})
                print(f"   ✅ Final Result:")
                print(f"      - Content Found: {final.get('content_found', False)}")
                print(f"      - Search Strategy: {final.get('search_strategy', 'unknown')}")
            
            # Save detailed response to file
            timestamp = int(time.time())
            filename = f"tracking_result_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n💾 Detailed tracking data saved to: {filename}")
            
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    """
    Run test cases to demonstrate tracking
    """
    print("🧪 CONTENT TRACKING SYSTEM TEST")
    print("This demonstrates how the system tracks content retrieval success/failure")
    
    # Test cases
    test_cases = [
        {
            "query": "define big data",
            "book_name": "Big_data .pdf",
            "description": "Query that should find content"
        },
        {
            "query": "what are the traditional database management techniques",
            "book_name": "Big_data .pdf", 
            "description": "Follow-up query that should find content"
        },
        {
            "query": "about gpu",
            "book_name": "Big_data .pdf",
            "description": "Query that should NOT find content"
        },
        {
            "query": "what is analytics",
            "book_name": "Big_data .pdf",
            "description": "Query that should NOT find content"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n🧪 TEST CASE {i}: {test_case['description']}")
        test_query_with_tracking(
            test_case["query"], 
            test_case["book_name"],
            f"test_session_{i}"
        )
        time.sleep(1)  # Small delay between requests
    
    print(f"\n\n✅ All tests completed!")
    print("Check the generated JSON files for detailed tracking data.")

if __name__ == "__main__":
    main() 