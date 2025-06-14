import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_signup() -> Dict[str, Any]:
    # Load environment variables
    load_dotenv()
    
    # Get Supabase credentials
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_ANON_KEY')
    
    if not url or not key:
        raise ValueError("Missing Supabase credentials")
    
    logger.info(f"Using Supabase URL: {url}")
    
    # Initialize Supabase client
    supabase: Client = create_client(url, key)
    
    # Test credentials
    email = "poker.ap4549@gmail.com"
    password = "VoiceBot@2025!"
    
    try:
        # Try sign in first
        logger.info("Attempting sign in...")
        try:
            auth_response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            logger.info("✅ Sign in successful!")
            user = auth_response.user
            logger.info(f"User ID: {user.id if user else 'No user ID'}")
            return auth_response
            
        except Exception as e:
            logger.info(f"Sign in failed, attempting registration: {str(e)}")
        
        # If sign in fails, try registration
        logger.info("Attempting registration...")
        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        
        logger.info(f"Registration response type: {type(auth_response)}")
        logger.info(f"Registration response: {auth_response}")
        
        if hasattr(auth_response, 'user') and auth_response.user:
            logger.info("✅ Registration successful!")
            logger.info(f"User ID: {auth_response.user.id}")
            logger.info(f"Email confirmation status: {auth_response.user.email_confirmed_at}")
            logger.info("\nIMPORTANT: Please check your email (including spam) for verification")
            logger.info("Email should come from noreply@mail.app.supabase.io")
            logger.info("You must verify your email before you can sign in")
        else:
            logger.error("❌ No user in registration response")
            
        return auth_response
            
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        if hasattr(e, 'response'):
            logger.error(f"Error response: {e.response}")
        raise

if __name__ == "__main__":
    test_signup()
