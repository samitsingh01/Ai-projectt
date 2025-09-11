#!/usr/bin/env python3
# intelligent-qa-service/startup.py - Service startup and initialization
import os
import sys
import logging
import time
import asyncio
from pathlib import Path

# Add the current directory to Python path for service imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'services',
        'temp'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Directory '{directory}' ready")

def check_environment():
    """Check required environment variables"""
    required_vars = [
        'DATABASE_URL',
        'BEDROCK_SERVICE_URL',
        'FILE_SERVICE_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Using default values for missing variables")
    else:
        logger.info("All required environment variables are set")

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import fastapi
        import mysql.connector
        import sklearn
        import numpy
        import httpx
        logger.info("All core dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def wait_for_dependencies():
    """Wait for dependent services to be ready"""
    import httpx
    import mysql.connector
    from mysql.connector import Error
    
    # Wait for database
    database_url = os.getenv("DATABASE_URL", "mysql://bedrock_user:bedrock_password@mysql:3306/bedrock_chat")
    max_retries = 30
    retry_count = 0
    
    logger.info("Waiting for database connection...")
    while retry_count < max_retries:
        try:
            url_parts = database_url.replace("mysql://", "").split("/")
            auth_host = url_parts[0].split("@")
            auth = auth_host[0].split(":")
            host_port = auth_host[1].split(":")
            
            connection = mysql.connector.connect(
                host=host_port[0],
                port=int(host_port[1]) if len(host_port) > 1 else 3306,
                user=auth[0],
                password=auth[1],
                database=url_parts[1],
                connection_timeout=5
            )
            connection.close()
            logger.info("✓ Database connection successful")
            break
        except Error as e:
            retry_count += 1
            logger.info(f"Database not ready (attempt {retry_count}/{max_retries}): {e}")
            time.sleep(2)
    
    if retry_count >= max_retries:
        logger.error("Failed to connect to database after maximum retries")
        return False
    
    # Wait for Bedrock service
    bedrock_url = os.getenv("BEDROCK_SERVICE_URL", "http://bedrock-service:9000")
    logger.info("Waiting for Bedrock service...")
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{bedrock_url}/health")
                if response.status_code == 200:
                    logger.info("✓ Bedrock service is ready")
                    break
        except Exception as e:
            retry_count += 1
            logger.info(f"Bedrock service not ready (attempt {retry_count}/{max_retries}): {e}")
            time.sleep(2)
    
    if retry_count >= max_retries:
        logger.warning("Bedrock service not available, but continuing startup")
    
    return True

def initialize_services():
    """Initialize and test services"""
    try:
        # Import and test services
        from services.document_chunker import DocumentChunker
        from services.vector_store import VectorStore
        from services.rag_engine import RAGEngine
        from services.citation_tracker import CitationTracker
        
        database_url = os.getenv("DATABASE_URL", "mysql://bedrock_user:bedrock_password@mysql:3306/bedrock_chat")
        bedrock_url = os.getenv("BEDROCK_SERVICE_URL", "http://bedrock-service:9000")
        
        # Initialize services
        document_chunker = DocumentChunker(max_chunk_size=1000)
        vector_store = VectorStore(database_url)
        rag_engine = RAGEngine(bedrock_url)
        citation_tracker = CitationTracker()
        
        logger.info("✓ All services initialized successfully")
        
        # Test document chunker
        test_text = "This is a test document. It has multiple sentences. This tests the chunking functionality."
        chunks = document_chunker.chunk_document(test_text, "test.txt")
        logger.info(f"✓ Document chunker test: created {len(chunks)} chunks")
        
        return True
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False

def run_startup_checks():
    """Run all startup checks"""
    logger.info("=" * 50)
    logger.info("🚀 Starting Intelligent Q&A Service")
    logger.info("=" * 50)
    
    # Step 1: Create directories
    logger.info("1. Creating directories...")
    create_directories()
    
    # Step 2: Check environment
    logger.info("2. Checking environment...")
    check_environment()
    
    # Step 3: Check dependencies
    logger.info("3. Checking dependencies...")
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        return False
    
    # Step 4: Wait for dependencies
    logger.info("4. Waiting for dependent services...")
    if not wait_for_dependencies():
        logger.error("❌ Dependent services not available")
        return False
    
    # Step 5: Initialize services
    logger.info("5. Initializing services...")
    if not initialize_services():
        logger.error("❌ Service initialization failed")
        return False
    
    logger.info("=" * 50)
    logger.info("✅ Startup checks completed successfully!")
    logger.info("🎯 Ready to process intelligent Q&A requests")
    logger.info("=" * 50)
    
    return True

if __name__ == "__main__":
    if run_startup_checks():
        # Start the main application
        logger.info("Starting FastAPI application...")
        import uvicorn
        from main import app
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=6000,
            log_level="info",
            access_log=True
        )
    else:
        logger.error("Startup checks failed. Exiting.")
        sys.exit(1)
