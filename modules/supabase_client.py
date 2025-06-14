from supabase import create_client, Client
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseManager:
    def __init__(self):
        """Initialize Supabase client with credentials from environment variables."""
        load_dotenv()
        
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials not found in environment variables")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("Supabase client initialized successfully")    
    def sign_up(self, email: str, password: str) -> Dict:
        """Register a new user."""
        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password
            })
            logger.info(f"User registered successfully: {email}")
            return response
        except Exception as e:
            logger.error(f"Error during sign up: {str(e)}")
            raise

    async def sign_in(self, email: str, password: str) -> Dict:
        """Sign in an existing user."""
        try:
            response = await self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            logger.info(f"User logged in successfully: {email}")
            return response
        except Exception as e:
            logger.error(f"Error during sign in: {str(e)}")
            raise

    async def sign_out(self) -> None:
        """Sign out the current user."""
        try:
            await self.client.auth.sign_out()
            logger.info("User signed out successfully")
        except Exception as e:
            logger.error(f"Error during sign out: {str(e)}")
            raise

    async def store_conversation(
        self,
        user_id: str,
        message: str,
        response: str,
        audio_url: Optional[str] = None,
        response_audio_url: Optional[str] = None,
        language: Optional[str] = None,
        confidence_score: Optional[float] = None,
        timestamp: Optional[str] = None
    ) -> Dict:
        """Store a conversation in the database."""
        try:
            data = {
                "user_id": user_id,
                "message": message,
                "response": response,
                "audio_url": audio_url,
                "response_audio_url": response_audio_url,
                "language": language,
                "confidence_score": confidence_score,
                "timestamp": timestamp
            }
            
            result = await self.client.table('conversations').insert(data).execute()
            logger.info(f"Conversation stored successfully for user {user_id}")
            return result.data[0]
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            raise

    async def get_recent_conversations(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """Retrieve recent conversations for a user."""
        try:
            response = await self.client.table('conversations')\
                .select("*")\
                .eq("user_id", user_id)\
                .order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error fetching conversations: {str(e)}")
            raise

    async def get_session(self) -> Optional[Dict]:
        """Get the current session if it exists."""
        try:
            session = await self.client.auth.get_session()
            return session if session else None
        except Exception as e:
            logger.error(f"Error getting session: {str(e)}")
            return None
