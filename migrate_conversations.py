import asyncio
import pandas as pd
from pathlib import Path
from modules.supabase_client import SupabaseManager
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate_existing_conversations():
    try:
        # Load existing conversations
        input_csv = Path("Data/userinputvoice/user_inputs.csv")
        responses_csv = Path("Data/voicebotoutput/voice_responses.csv")
        
        inputs_df = pd.read_csv(input_csv)
        responses_df = pd.read_csv(responses_csv)
        
        logger.info(f"Found {len(responses_df)} existing conversations to migrate")
        
        # Initialize Supabase
        supabase = SupabaseManager()
        logger.info("✅ Successfully initialized Supabase client")
        
        # Create a test user for existing conversations
        try:
            response = await supabase.sign_up(
                email="voicebot.test@example.com",
                password="VoiceBot@2025!"
            )
            user_id = response.user.id if response.user else None
            logger.info(f"Created new user with ID: {user_id}")
        except Exception as e:
            # Try logging in if user exists
            try:
                response = await supabase.sign_in(
                    email="voicebot.test@example.com",
                    password="VoiceBot@2025!"
                )
                user_id = response.user.id if response.user else None
                logger.info(f"Logged in existing user with ID: {user_id}")
            except Exception as login_error:
                logger.error(f"Could not authenticate: {login_error}")
                return
        
        if not user_id:
            logger.error("No user ID available for migration")
            return
        
        # Migrate each conversation
        for _, row in responses_df.iterrows():
            try:
                # Store conversation in Supabase
                result = await supabase.store_conversation(
                    user_id=user_id,
                    message=row['Question'],
                    response=row['Response'],
                    audio_url=inputs_df[inputs_df['Question'] == row['Question']]['AudioFilePath'].iloc[0],
                    language=row['Language'],
                    confidence_score=float(row['Confidence']),
                    timestamp=row['Timestamp']
                )
                logger.info(f"✅ Migrated conversation: {row['Question'][:50]}...")
            except Exception as e:
                logger.error(f"Failed to migrate conversation: {str(e)}")
        
        # Test retrieval
        conversations = await supabase.get_recent_conversations(user_id)
        logger.info(f"✅ Successfully retrieved {len(conversations)} conversations")
        
        # Print first conversation as sample
        if conversations:
            logger.info("Sample conversation:")
            logger.info(f"Q: {conversations[0].get('message')}")
            logger.info(f"A: {conversations[0].get('response')}")
    
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(migrate_existing_conversations())
