// Test script to verify backend endpoints
const BACKEND_URL = 'http://localhost:8000'; // Change this to your backend URL

async function testBackend() {
  console.log('🧪 Testing Backend Endpoints...\n');
  
  try {
    // Test 1: Health Check
    console.log('1️⃣ Testing Health Check...');
    const healthResponse = await fetch(`${BACKEND_URL}/health`);
    const healthData = await healthResponse.json();
    console.log('✅ Health Check:', healthData);
    
    // Test 2: Root Endpoint
    console.log('\n2️⃣ Testing Root Endpoint...');
    const rootResponse = await fetch(`${BACKEND_URL}/`);
    const rootData = await rootResponse.json();
    console.log('✅ Root Endpoint:', rootData);
    
    // Test 3: Get Documents
    console.log('\n3️⃣ Testing Get Documents...');
    const docsResponse = await fetch(`${BACKEND_URL}/documents`);
    const docsData = await docsResponse.json();
    console.log('✅ Documents:', docsData);
    
    // Test 4: Get Chunks
    console.log('\n4️⃣ Testing Get Chunks...');
    const chunksResponse = await fetch(`${BACKEND_URL}/chunks`);
    const chunksData = await chunksResponse.json();
    console.log('✅ Chunks:', chunksData);
    
    console.log('\n🎉 All tests passed! Backend is working correctly.');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.log('\n🔧 Troubleshooting tips:');
    console.log('1. Make sure your backend is running on the correct port');
    console.log('2. Check if the backend URL is correct');
    console.log('3. Verify Google Cloud credentials are set up');
    console.log('4. Check backend logs for errors');
  }
}

// Run the test
testBackend(); 