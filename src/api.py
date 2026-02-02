"""
FastAPI Backend - Serve static files and API endpoints
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import uuid
from datetime import datetime
import yfinance as yf
import os
import shutil
import PyPDF2
from docx import Document

from src.agents.research_agent import ResearchAgent
from src.core.report_generator import ReportGenerator
from src.utils.logger import get_logger

logger = get_logger("api")

# Initialize FastAPI
app = FastAPI(
    title="Financial Research AI",
    description="AI-powered financial research assistant",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global instances
research_agent: Optional[ResearchAgent] = None
report_generator: Optional[ReportGenerator] = None
chat_sessions: Dict[str, List[Dict]] = {}

# Models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    session_id: str
    timestamp: str
    report_available: bool
    report_path: Optional[str] = None

class StockQuery(BaseModel):
    symbol: str
    period: str = "1mo"

# Startup
@app.on_event("startup")
async def startup_event():
    global research_agent, report_generator
    
    logger.info("Starting Financial Research AI API...")
    
    try:
        research_agent = ResearchAgent()
        report_generator = ReportGenerator()
        
        # Create directories
        Path("outputs").mkdir(exist_ok=True)
        Path("uploads").mkdir(exist_ok=True)
        
        logger.info("API initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize", error=e)
        raise

# Routes
@app.get("/")
async def root():
    """Serve main page"""
    return FileResponse("static/index.html")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat messages"""
    if not research_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        session_id = message.session_id or str(uuid.uuid4())
        logger.info(f"Chat request: {message.message[:50]}...")
        
        # Research
        result = await research_agent.research(
            query=message.message,
            use_web=True
        )
        
        # Generate report
        report_path = None
        if report_generator:
            report_path = report_generator.save_report(result, format="html")
        
        # Store history
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_sessions[session_id].extend([
            {'role': 'user', 'content': message.message, 'timestamp': datetime.now().isoformat()},
            {'role': 'assistant', 'content': result.answer, 'timestamp': datetime.now().isoformat()}
        ])
        
        return ChatResponse(
            response=result.answer,
            sources=result.sources,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            report_available=report_path is not None,
            report_path=report_path
        )
        
    except Exception as e:
        logger.error("Chat failed", error=e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/price/{symbol}")
async def get_stock_price(symbol: str):
    """Get current stock price"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        change = info.get('regularMarketChangePercent', 0)
        
        return {
            "symbol": symbol,
            "price": current_price,
            "change": change
        }
    except Exception as e:
        logger.error(f"Stock price error for {symbol}", error=e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/stock/chart")
async def get_stock_chart(query: StockQuery):
    """Get stock chart data"""
    try:
        stock = yf.Ticker(query.symbol)
        hist = stock.history(period=query.period)
        
        return {
            "symbol": query.symbol,
            "dates": hist.index.strftime('%Y-%m-%d').tolist(),
            "prices": hist['Close'].tolist(),
            "volume": hist['Volume'].tolist()
        }
    except Exception as e:
        logger.error(f"Chart error for {query.symbol}", error=e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process documents (PDF, DOCX, TXT)"""
    try:
        logger.info(f"Receiving file upload: {file.filename}")
        
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt', '.doc']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save file
        upload_path = Path("uploads") / file.filename
        upload_path.parent.mkdir(exist_ok=True)
        
        file_content = await file.read()
        with upload_path.open("wb") as buffer:
            buffer.write(file_content)
        
        # Extract text based on file type
        text_content = ""
        
        if file_ext == '.pdf':
            with open(upload_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
        
        elif file_ext in ['.docx', '.doc']:
            doc = Document(upload_path)
            text_content = "\n".join([para.text for para in doc.paragraphs])
        
        elif file_ext == '.txt':
            with open(upload_path, 'r', encoding='utf-8') as txt_file:
                text_content = txt_file.read()
        
        logger.info(f"Extracted {len(text_content)} characters from {file.filename}")
        
        return {
            "success": True,
            "filename": file.filename,
            "file_path": str(upload_path),
            "text_length": len(text_content),
            "preview": text_content[:500] + "..." if len(text_content) > 500 else text_content,
            "message": "File uploaded and processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-document")
async def analyze_document(request: dict):
    """Analyze uploaded document with a specific query"""
    try:
        file_path = request.get('file_path')
        query = request.get('query', 'Summarize this document')
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Read document content
        file_ext = os.path.splitext(file_path)[1].lower()
        content = ""
        
        if file_ext == '.pdf':
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_ext in ['.docx', '.doc']:
            doc = Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
        
        # Use research agent to analyze
        result = await research_agent.research(
            query=f"{query}\n\nDocument Content:\n{content[:10000]}"
        )
        
        return {
            "success": True,
            "query": query,
            "analysis": result.get('response', ''),
            "sources": result.get('sources', [])
        }
        
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
async def list_files():
    """List all uploaded files"""
    try:
        upload_dir = Path("uploads")
        if not upload_dir.exists():
            return {"files": []}
        
        files = []
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return {"files": files}
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/{filename}")
async def get_report(filename: str):
    """Download report"""
    report_path = Path("outputs") / filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path)

@app.get("/api/status")
async def get_status():
    """API status"""
    return {
        "status": "ready" if research_agent else "initializing",
        "version": "3.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
