"""
MongoDB Storage Adapter for Transcription Agent

Sends transcripts to chat-agent's MongoDB via HTTP API instead of local filesystem.
"""
import os
import json
import uuid
from typing import Optional, Dict, Any
import httpx

class MongoDBStore:
    """Store transcripts in MongoDB via chat-agent API"""
    
    @staticmethod
    def new_transcript_id() -> str:
        """Generate new transcript ID"""
        return "t_" + uuid.uuid4().hex[:16]
    
    @staticmethod
    async def save_transcript(
        transcript_id: str,
        payload: dict,
        org_id: str | None = None,
        meeting_id: str | None = None
    ) -> str:
        """
        Save transcript to MongoDB via chat-agent API
        
        Args:
            transcript_id: Unique transcript identifier
            payload: Transcript data (text, segments, metadata)
            org_id: Organization ID
            meeting_id: Meeting ID (optional)
            
        Returns:
            transcript_id on success
            
        Raises:
            RuntimeError: If API call fails
        """
        chat_agent_url = os.getenv('CHAT_AGENT_URL', 'http://localhost:3000')
        service_token = os.getenv('SERVICE_TOKEN')
        
        # Prepare bundle for ingestion
        bundle = {
            'bundle': {
                'transcript': {
                    'raw_text': payload.get('text', ''),
                    'text': payload.get('text', ''),
                    'utterances': payload.get('segments', []),
                    'notes': payload.get('notes', {}),
                    'summary': payload.get('summary', ''),
                    'index': payload.get('index', {})
                },
                'source': payload.get('source', 'transcription_agent'),
                'facts': []  # Facts will be added by parsing-agent
            },
            'transcript_id': transcript_id,
            'org_id': org_id or 'org_demo',
            'meeting_id': meeting_id,
            'status': 'draft'
        }
        
        headers = {'Content-Type': 'application/json'}
        if service_token:
            headers['Authorization'] = f'Bearer {service_token}'
        
        endpoint = f'{chat_agent_url.rstrip("/")}/api/spine/bundles/ingest'
        
        try:
            print(f'[TA][MongoDB] Saving transcript {transcript_id} to {endpoint}')
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(endpoint, json=bundle, headers=headers)
                
                if response.status_code >= 400:
                    error_text = response.text[:500]
                    raise RuntimeError(
                        f'MongoDB store failed: HTTP {response.status_code} - {error_text}'
                    )
                
                result = response.json()
                print(f'[TA][MongoDB] Saved transcript {transcript_id}: {result}')
                
                return transcript_id
                
        except httpx.RequestError as e:
            raise RuntimeError(f'MongoDB store request failed: {str(e)}')
        except Exception as e:
            raise RuntimeError(f'MongoDB store error: {str(e)}')
    
    @staticmethod
    async def load_transcript_json(transcript_id: str) -> Optional[dict]:
        """
        Load transcript from MongoDB via chat-agent API
        
        Args:
            transcript_id: Transcript ID to load
            
        Returns:
            Transcript dict or None if not found
        """
        chat_agent_url = os.getenv('CHAT_AGENT_URL', 'http://localhost:3000')
        service_token = os.getenv('SERVICE_TOKEN')
        
        headers = {}
        if service_token:
            headers['Authorization'] = f'Bearer {service_token}'
        
        endpoint = f'{chat_agent_url.rstrip("/")}/api/spine/transcripts/{transcript_id}/download'
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(endpoint, headers=headers)
                
                if response.status_code == 404:
                    return None
                
                if response.status_code >= 400:
                    print(f'[TA][MongoDB] Load failed: HTTP {response.status_code}')
                    return None
                
                # API returns plain text, wrap it in a dict
                text = response.text
                return {'text': text, 'transcript_id': transcript_id}
                
        except Exception as e:
            print(f'[TA][MongoDB] Load error: {str(e)}')
            return None
    
    @staticmethod
    async def load_transcript_text(transcript_id: str) -> Optional[str]:
        """
        Load transcript text from MongoDB
        
        Args:
            transcript_id: Transcript ID
            
        Returns:
            Plain text or None
        """
        data = await MongoDBStore.load_transcript_json(transcript_id)
        return data.get('text') if data else None
