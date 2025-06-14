import asyncio
from modules.supabase_client import SupabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_supabase():
    try:
        # Initialize Supabase manager
        supabase = SupabaseManager()
        logger.info("✅ Successfully initialized Supabase client")

        # Test user credentials
        email = "poker.ap4549@gmail.com"
        password = "VoiceBot@2025!"        # First, sign in to get a valid session
        try:
            response = supabase.sign_in(email, password)
            user_id = response.user.id if response.user else None
            if not user_id:
                logger.error("❌ Failed to get user ID after login")
                return
            logger.info(f"✅ Successfully logged in user: {email}")
            logger.info(f"User ID: {user_id}")
            
            # Store test conversation
            result = await supabase.store_conversation(
                user_id=user_id,
                message="This is a test message from the voice bot.",
                response="This is a test response from the system.",
                language="en",
                confidence_score=1.0
            )
            logger.info("✅ Test conversation stored successfully!")
            
            # Retrieve conversations
            conversations = await supabase.get_recent_conversations(user_id)
            logger.info(f"✅ Retrieved {len(conversations)} conversations")
            if conversations:
                logger.info("Sample conversation:")
                logger.info(f"Message: {conversations[0]['message']}")
                logger.info(f"Response: {conversations[0]['response']}")
            
        except Exception as conv_error:
            logger.error(f"❌ Conversation test failed: {str(conv_error)}")

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_supabase())
