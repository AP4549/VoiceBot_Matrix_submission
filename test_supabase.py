import asyncio
from modules.supabase_client import SupabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_supabase_connection():
    try:
        # Initialize Supabase manager
        supabase = SupabaseManager()
        logger.info("✅ Successfully initialized Supabase client")

        # Test registration
        try:
            response = await supabase.sign_up("test@example.com", "testpassword123")
            logger.info("✅ Registration test successful")
            logger.info(f"User ID: {response.user.id if response.user else 'None'}")
        except Exception as e:
            logger.error(f"❌ Registration test failed: {str(e)}")

        # Test login
        try:
            response = await supabase.sign_in("test@example.com", "testpassword123")
            logger.info("✅ Login test successful")
            logger.info(f"User ID: {response.user.id if response.user else 'None'}")
        except Exception as e:
            logger.error(f"❌ Login test failed: {str(e)}")

        # Test conversation storage
        try:
            user_id = response.user.id
            result = await supabase.store_conversation(
                user_id=user_id,
                message="Hello, this is a test message",
                response="This is a test response",
                language="en",
                confidence_score=0.95
            )
            logger.info("✅ Conversation storage test successful")
            logger.info(f"Stored conversation ID: {result.get('id')}")
        except Exception as e:
            logger.error(f"❌ Conversation storage test failed: {str(e)}")

        # Test conversation retrieval
        try:
            conversations = await supabase.get_recent_conversations(user_id)
            logger.info("✅ Conversation retrieval test successful")
            logger.info(f"Retrieved {len(conversations)} conversations")
            for conv in conversations:
                logger.info(f"Conversation: {conv.get('message')} -> {conv.get('response')}")
        except Exception as e:
            logger.error(f"❌ Conversation retrieval test failed: {str(e)}")

    except Exception as e:
        logger.error(f"❌ Supabase connection test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_supabase_connection())
