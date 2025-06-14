import asyncio
from modules.supabase_client import SupabaseManager
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_test_user():
    try:
        # Initialize Supabase manager
        supabase = SupabaseManager()
        logger.info("✅ Successfully initialized Supabase client")

        # Test user credentials
        email = "poker.ap4549@gmail.com"
        password = "VoiceBot@2025!"  # Strong password for testing        # Try registration
        try:
            logger.info(f"Attempting to register user: {email}")
            response = supabase.sign_up(email, password)
            logger.info(f"Registration response: {response}")
            
            if hasattr(response, 'user') and response.user:
                logger.info(f"✅ User created with ID: {response.user.id}")
                if hasattr(response.user, 'confirmation_sent_at'):
                    logger.info(f"Confirmation email sent at: {response.user.confirmation_sent_at}")
                else:
                    logger.warning("No confirmation_sent_at timestamp found")
            else:
                logger.warning("Response doesn't contain user data")
            
            logger.info("Please check your email (including spam folder) for verification link")
            logger.info("Email should come from noreply@mail.app.supabase.io")
            input("Press Enter after verifying your email...")
            
        except Exception as reg_error:
            if "User already registered" in str(reg_error):
                logger.info("User already exists, proceeding to sign in")
            else:
                logger.error(f"❌ Registration failed with error: {str(reg_error)}")
                logger.error(f"Error type: {type(reg_error)}")
                if hasattr(reg_error, 'response'):
                    logger.error(f"Error response: {reg_error.response}")
                raise

        # Sign in and get session
        try:
            response = await supabase.sign_in(email, password)
            session = await supabase.get_session()
            if not session or not session.user:
                raise Exception("No session or user found after login")
            
            user_id = session.user.id
            logger.info("✅ Login successful!")
            logger.info(f"User ID: {user_id}")

            # Save user_id to .env if not exists
            env_path = ".env"
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    env_content = f.read()
            else:
                env_content = ""
            
            if "SUPABASE_USER_ID" not in env_content:
                with open(env_path, "a") as f:
                    f.write(f"\nSUPABASE_USER_ID={user_id}")
                logger.info("Added SUPABASE_USER_ID to .env file")

        except Exception as login_error:
            logger.error(f"❌ Login failed: {str(login_error)}")
            raise

        # Test conversation storage if we have a user_id
        try:
            result = await supabase.store_conversation(
                user_id=user_id,
                message="This is a test message from the voice bot.",
                response="This is a test response from the system.",
                language="en",
                confidence_score=1.0,
                format="text",
                source="test"
            )
            logger.info("✅ Test conversation stored successfully!")
            
            # Retrieve conversations
            conversations = await supabase.get_recent_conversations(user_id)
            logger.info(f"✅ Retrieved {len(conversations)} conversations")
            
        except Exception as conv_error:
            logger.error(f"❌ Conversation test failed: {str(conv_error)}")
            raise

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(create_test_user())
