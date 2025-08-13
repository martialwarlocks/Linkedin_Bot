// Test content cleaning logic
const testLinkedinPost = '"AI: The Future of Human-Machine Collaboration\nAs AI continues to transform industries, it\'s essential to understand its potential to augment human capabilities rather than replace them. By embracing AI-driven tools, we can unlock new levels of productivity, creativity, and innovation.\nBut what does this mean for the future of work? Will AI create more jobs or replace them? The answer lies in our ability to adapt and collaborate with machines.\nJoin me in exploring the exciting possibilities of AI-human collaboration and how it can shape the future of your industry. #AI #FutureOfWork #Collaboration #Innovation"';

const testVideoScript = '"(0:00)\nHey everyone, welcome back to my channel! Today, I want to talk about AI and how it\'s changing the game.\n[0:05]\nNow, I know some of you might be thinking, \"AI is going to replace my job!\" But the truth is, AI is not meant to replace us, it\'s meant to augment our capabilities.\n[0:15]\nThink about it, AI can help us automate repetitive tasks, freeing up our time to focus on the creative and strategic work that really matters. And with AI-driven tools, we can make decisions faster and more accurately than ever before.\n[0:30]\nBut the real magic happens when we combine human intuition with AI\'s analytical power. That\'s when we unlock new levels of innovation and creativity.\n[0:40]\nSo, what does this mean for the future of work? Well, I believe AI will create more jobs than it replaces. The key is to adapt and collaborate with machines.\n[0:50]\nSo, are you ready to take the leap and start exploring the possibilities of AI-human collaboration? Let me know in the comments!"';

function cleanContent(text) {
  let cleaned = text;
  
  // Remove outer quotes if they exist
  if (cleaned.startsWith('"') && cleaned.endsWith('"')) {
    cleaned = cleaned.slice(1, -1);
  }
  
  // Also remove any remaining quotes and clean up
  cleaned = cleaned.replace(/^["']+|["']+$/g, '').trim();
  
  return cleaned;
}

console.log('=== TESTING CONTENT CLEANING ===');
console.log('Original LinkedIn post length:', testLinkedinPost.length);
console.log('Original video script length:', testVideoScript.length);

const cleanedLinkedinPost = cleanContent(testLinkedinPost);
const cleanedVideoScript = cleanContent(testVideoScript);

console.log('Cleaned LinkedIn post length:', cleanedLinkedinPost.length);
console.log('Cleaned video script length:', cleanedVideoScript.length);
console.log('LinkedIn post starts with quote?', cleanedLinkedinPost.startsWith('"'));
console.log('Video script starts with quote?', cleanedVideoScript.startsWith('"'));
console.log('LinkedIn post first 100 chars:', cleanedLinkedinPost.substring(0, 100));
console.log('Video script first 100 chars:', cleanedVideoScript.substring(0, 100)); 