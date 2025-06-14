import os
import pandas as pd
from dotenv import load_dotenv
from modules.supabase_client import SupabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_conversations(user_id: str):
    """Migrate conversations from CSV files to Supabase for a given user."""
    try:
        # Initialize Supabase client
        supabase = SupabaseManager()
        logger.info(f"Initialized Supabase client with URL: {supabase.supabase_url}")
        
        # Sign in to get access token
        logger.info("Authenticating with Supabase...")
        try:
            auth_response = supabase.sign_in("poker.ap4549@gmail.com", "VoiceBot@2025!")
            logger.info(f"Auth response type: {type(auth_response)}")
            logger.info(f"Auth response attributes: {dir(auth_response)}")
            
            if hasattr(auth_response, 'session'):
                session = auth_response.session
                logger.info(f"Session type: {type(session)}")
                logger.info(f"Session attributes: {dir(session)}")
                access_token = session.access_token
                logger.info("✅ Got access token")
            else:
                raise ValueError("No session in auth response")
                
        except Exception as auth_error:
            logger.error(f"Authentication failed: {str(auth_error)}")
            if hasattr(auth_error, 'message'):
                logger.error(f"Auth error message: {auth_error.message}")
            raise
        
        # Read CSV files
        try:
            inputs_df = pd.read_csv('Data/userinputvoice/user_inputs.csv')
            responses_df = pd.read_csv('Data/voicebotoutput/voice_responses.csv')
            logger.info(f"Read input CSV ({len(inputs_df)} rows)")
            logger.info(f"Read output CSV ({len(responses_df)} rows)")
        except Exception as csv_error:
            logger.error(f"Failed to read CSV files: {str(csv_error)}")
            raise
        
        # Merge the dataframes on Timestamp and Question
        try:
            merged_df = pd.merge(inputs_df, responses_df, 
                               on=['Timestamp', 'Question'],
                               suffixes=('_input', '_response'))
            total_rows = len(merged_df)
            logger.info(f"Merged CSVs - found {total_rows} conversations to migrate")
        except Exception as merge_error:
            logger.error(f"Failed to merge dataframes: {str(merge_error)}")
            raise
        
        # Migrate each conversation
        for idx, row in merged_df.iterrows():
            try:
                logger.info(f"\nProcessing conversation {idx + 1}/{total_rows}:")
                logger.info(f"Question: {row['Question'][:50]}...")
                
                result = supabase.store_conversation(
                    user_id=user_id,
                    message=row['Question'],
                    response=row['Response'],
                    audio_url=row['AudioFilePath'] if pd.notna(row['AudioFilePath']) else None,
                    response_audio_url=None,  # We don't store this in CSV currently
                    language=row['Language_input'],
                    confidence_score=row.get('Confidence', None),
                    format=row['Format_input'],
                    source=row.get('Source', None),
                    access_token=access_token
                )
                logger.info(f"✅ Successfully migrated conversation {idx + 1}")
            except Exception as e:
                logger.error(f"❌ Error migrating conversation {idx + 1}:")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error string: {str(e)}")
                if hasattr(e, 'message'):
                    logger.error(f"Error message: {e.message}")
                if hasattr(e, 'response'):
                    logger.error(f"Error response: {e.response}")
                raise
        
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    load_dotenv()
    
    # Get user_id from environment or input
    user_id = os.getenv("SUPABASE_USER_ID")
    if not user_id:
        raise ValueError("SUPABASE_USER_ID environment variable is required")
    
    logger.info(f"Starting migration for user ID: {user_id}")
    migrate_conversations(user_id)
