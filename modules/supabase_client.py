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
            logger.info(f"Attempting to register user with email: {email}")
            response = self.client.auth.sign_up({
                "email": email,
                "password": password
            })
            logger.info(f"Raw signup response: {response}")
            
            if response and hasattr(response, 'user'):
                logger.info(f"User registered successfully: {email}")
                return response
            else:
                raise ValueError(f"Invalid signup response format: {response}")
                
        except Exception as e:
            logger.error(f"Error during sign up: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Error response details: {e.response}")
            raise    
    def sign_in(self, email: str, password: str) -> Dict:
        """Sign in an existing user."""
        try:
            logger.info(f"Attempting to sign in user: {email}")
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            if hasattr(response, 'session') and response.session:
                logger.info(f"✅ User logged in successfully: {email}")
                logger.info(f"Session token type: {type(response.session)}")
                logger.info(f"Access token obtained: {bool(response.session.access_token)}")
            else:
                logger.warning("No session in response")
                logger.info(f"Response type: {type(response)}")
                logger.info(f"Response attributes: {dir(response)}")
            return response
        except Exception as e:
            logger.error(f"❌ Error during sign in: {str(e)}")
            if hasattr(e, 'message'):
                logger.error(f"Error message: {e.message}")
            raise
            
    def sign_out(self) -> None:
        """Sign out the current user."""
        try:
            self.client.auth.sign_out()
            logger.info("User signed out successfully")
        except Exception as e:
            logger.error(f"Error during sign out: {str(e)}")
            raise

    def store_conversation(
        self,
        user_id: str,
        message: str,
        response: str,
        audio_url: str = None,
        response_audio_url: str = None,
        language: str = None,
        confidence_score: float = None,
        format: str = None,
        source: str = None,
        access_token: str = None
    ) -> Dict:
        """Store a conversation in the database."""
        try:
            # First sign in if needed
            if not access_token:
                try:
                    auth_response = self.client.auth.sign_in_with_password({
                        "email": "poker.ap4549@gmail.com",
                        "password": "VoiceBot@2025!"
                    })
                    if auth_response and auth_response.session:
                        access_token = auth_response.session.access_token
                    else:
                        raise ValueError("Could not get access token")
                except Exception as auth_error:
                    logger.error(f"Authentication failed: {str(auth_error)}")
                    raise
            
            # Set up the request with auth
            data = {
                "user_id": user_id,
                "message": message,
                "response": response,
                "audio_url": audio_url,
                "response_audio_url": response_audio_url,
                "language": language,
                "confidence_score": confidence_score,
                "format": format,
                "source": source
            }
            
            logger.info(f"Storing conversation for user {user_id}")
              # Try to use the existing client with auth from sign in
            try:
                result = self.client.table('conversations').insert(data).execute()
                logger.info("Conversation stored successfully")
                return result.data[0] if result.data else None
            except Exception as e:
                logger.error(f"Error with first attempt: {str(e)}")
                
                # If that fails, try with a fresh client
                auth_client = create_client(
                    self.supabase_url,
                    self.supabase_key,
                )
                auth_client.postgrest.auth(access_token)
                
                logger.info("Making authenticated request to insert conversation")
                result = auth_client.table('conversations').insert(data).execute()
                logger.info("Conversation stored successfully")
                return result.data[0] if result.data else None
            
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            if hasattr(e, 'message'):
                logger.error(f"Error message: {e.message}")
            if hasattr(e, 'response'):
                logger.error(f"Error response: {e.response}")
            raise

    def get_recent_conversations(
        self,
        user_id: str,
        access_token: str = None,
        limit: int = 5
    ) -> List[Dict]:
        """Get recent conversations for a user."""
        try:
            logger.info(f"Fetching recent conversations for user {user_id}")
            
            # Create a fresh client with auth if token provided
            if access_token:
                auth_client = create_client(
                    self.supabase_url,
                    self.supabase_key,
                )
                auth_client.postgrest.auth(access_token)
                query = auth_client
            else:
                query = self.client
            
            # Query conversations table
            result = query.table('conversations')\
                .select('*')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()

            conversations = result.data if result and hasattr(result, 'data') else []
            logger.info(f"Found {len(conversations)} recent conversations")
            return conversations

        except Exception as e:
            logger.error(f"Error fetching conversations: {str(e)}")
            if hasattr(e, 'message'):
                logger.error(f"Error message: {e.message}")
            raise

    async def get_session(self) -> Optional[Dict]:
        """Get the current session if it exists."""
        try:
            session = await self.client.auth.get_session()
            return session if session else None
        except Exception as e:
            logger.error(f"Error getting session: {str(e)}")
            return None
        
    def store_conversation_with_embedding(
        self,
        user_id: str,
        message: str,
        response: str,
        embedding: list,
        metadata: dict = None,
        audio_url: str = None,
        response_audio_url: str = None,
        language: str = None,
        confidence_score: float = None,
        format: str = None,
        source: str = None,
        access_token: str = None
    ) -> Dict:
        """Store a conversation with embedding vector in the database."""
        try:
            # First sign in if needed
            if not access_token:
                try:
                    auth_response = self.client.auth.sign_in_with_password({
                        "email": "poker.ap4549@gmail.com",
                        "password": "VoiceBot@2025!"
                    })
                    if auth_response and auth_response.session:
                        access_token = auth_response.session.access_token
                    else:
                        raise ValueError("Could not get access token")
                except Exception as auth_error:
                    logger.error(f"Authentication failed: {str(auth_error)}")
                    raise
            
            # Set up the request with auth
            # Combine message and response for better semantic retrieval
            combined_text = f"Human: {message}\nAssistant: {response}"
            
            data = {
                "user_id": user_id,
                "message": message,
                "response": response,
                "combined_text": combined_text,
                "embedding": embedding,
                "metadata": metadata or {},
                "audio_url": audio_url,
                "response_audio_url": response_audio_url,
                "language": language,
                "confidence_score": confidence_score,
                "format": format,
                "source": source
            }
            
            logger.info(f"Storing conversation with embedding for user {user_id}")
              
            try:
                # Use the 'conversation_embeddings' table instead of 'conversations'
                result = self.client.table('conversation_embeddings').insert(data).execute()
                logger.info("Conversation with embedding stored successfully")
                return result.data[0] if result.data else None
            except Exception as e:
                logger.error(f"Error with first attempt: {str(e)}")
                
                # If that fails, try with a fresh client
                auth_client = create_client(
                    self.supabase_url,
                    self.supabase_key,
                )
                auth_client.postgrest.auth(access_token)
                
                logger.info("Making authenticated request to insert conversation with embedding")
                result = auth_client.table('conversation_embeddings').insert(data).execute()
                logger.info("Conversation with embedding stored successfully")
                return result.data[0] if result.data else None
            
        except Exception as e:
            logger.error(f"Error storing conversation with embedding: {str(e)}")
            if hasattr(e, 'message'):
                logger.error(f"Error message: {e.message}")
            if hasattr(e, 'response'):
                logger.error(f"Error response: {e.response}")
            raise
            
    def get_relevant_conversations(
        self,
        user_id: str,
        query_embedding: list,
        match_threshold: float = 0.6,
        match_count: int = 5,
        access_token: str = None
    ) -> List[Dict]:
        """Get semantically relevant conversations using vector similarity."""
        try:
            logger.info(f"Fetching relevant conversations for user {user_id}")
            
            # Create a fresh client with auth if token provided
            if access_token:
                auth_client = create_client(
                    self.supabase_url,
                    self.supabase_key,
                )
                auth_client.postgrest.auth(access_token)
                query = auth_client
            else:
                query = self.client
                
            # Use RPC function to find similar conversations
            # This assumes you've set up a similarity search function in Supabase
            result = query.rpc(
                'match_conversations', 
                {
                    'query_embedding': query_embedding,
                    'match_threshold': match_threshold,
                    'match_count': match_count,
                    'p_user_id': user_id
                }
            ).execute()

            conversations = result.data if result and hasattr(result, 'data') else []
            logger.info(f"Found {len(conversations)} relevant conversations")
            return conversations

        except Exception as e:
            logger.error(f"Error fetching relevant conversations: {str(e)}")
            if hasattr(e, 'message'):
                logger.error(f"Error message: {e.message}")
            return []
