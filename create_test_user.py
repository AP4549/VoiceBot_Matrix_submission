from modules.supabase_client import SupabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_user():
    try:
        # Initialize Supabase manager
        supabase = SupabaseManager()
        logger.info("✅ Successfully initialized Supabase client")

        # Test user credentials
        email = "poker.ap4549@gmail.com"
        password = "VoiceBot@2025!"  # Strong password for testing

        # Try registration
        try:
            response = supabase.sign_up(email, password)
            user_id = response.user.id if response.user else None
            logger.info("✅ Registration successful!")
            logger.info(f"User ID: {user_id}")
            logger.info("Please check your email for verification link")
            
        except Exception as reg_error:
            if "User already registered" in str(reg_error):
                # Try logging in if user exists
                try:
                    response = supabase.sign_in(email, password)
                    user_id = response.user.id if response.user else None
                    logger.info("✅ Login successful!")
                    logger.info(f"User ID: {user_id}")
                except Exception as login_error:
                    logger.error(f"❌ Login failed: {str(login_error)}")
            else:
                logger.error(f"❌ Registration failed: {str(reg_error)}")

        # Test conversation storage if we have a user_id
        if 'user_id' in locals() and user_id:
            try:
                result = supabase.store_conversation(
                    user_id=user_id,
                    message="This is a test message from the voice bot.",
                    response="This is a test response from the system.",
                    language="en",
                    confidence_score=1.0
                )
                logger.info("✅ Test conversation stored successfully!")
                
                # Retrieve conversations
                conversations = supabase.get_recent_conversations(user_id)
                logger.info(f"✅ Retrieved {len(conversations)} conversations")
                
            except Exception as conv_error:
                logger.error(f"❌ Conversation test failed: {str(conv_error)}")

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    create_test_user()
