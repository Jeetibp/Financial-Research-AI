# AI Agent System - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Agent Architecture](#agent-architecture)
3. [Agents Implemented](#agents-implemented)
4. [AI Models Used](#ai-models-used)
5. [APIs and Integrations](#apis-and-integrations)
6. [Tools and Utilities](#tools-and-utilities)
7. [API Keys Required](#api-keys-required)
8. [Agent Workflows](#agent-workflows)
9. [Setup and Configuration](#setup-and-configuration)

---

## 1. Project Overview

This is a **Multi-Agent AI System** for financial research that orchestrates multiple specialized agents to perform complex research, analysis, and content creation tasks. The system uses various AI models and APIs to gather, process, and synthesize information from multiple sources.

### Key Features:
- Multi-agent collaboration system
- Web research and data gathering
- Financial data analysis
- Content generation and reporting
- Real-time information retrieval
- Automated workflow orchestration
- Enhanced PDF processing (text, tables, charts)
- Deep research mode with multi-step planning

---

## 2. Agent Architecture

### Framework: **Custom Agent System** (Not CrewAI)
The project uses a custom-built agent system with specialized agents for different domains. Agents work together through:
- **Query classification and routing**
- **Specialized domain expertise**
- **Collaborative problem-solving**
- **Sequential and parallel workflows**

### Architecture Pattern:
```
User Input → Query Classifier → Router → Specialized Agent → Tools → External APIs → Results
```

### Core Components:
1. **Query Classifier** - Determines query complexity (INSTANT/SIMPLE/COMPLEX/DEEP)
2. **Query Router** - Routes to appropriate specialized agent
3. **Specialized Agents** - Domain-specific agents (IT Sector, Pharma Sector, General Research)
4. **Orchestrator** - Manages multi-agent workflows
5. **Deep Research Executor** - Handles complex multi-step research

---

## 3. Agents Implemented

Based on the project structure, the following **specialized agents** are implemented:

### 3.1 **Base Agent** (`base_agent.py`)
- **Role**: Abstract base class for all agents
- **Purpose**: Provides common functionality and interface
- **Capabilities**:
  - Standard query processing
  - Tool integration
  - Response formatting

---

### 3.2 **Research Agent** (`research_agent.py`)
- **Role**: General-purpose research and information gathering
- **Goal**: Conduct comprehensive research on any topic
- **Capabilities**:
  - Web search integration (Tavily API)
  - Web scraping
  - Vector store retrieval (RAG)
  - Multi-source synthesis
  - Financial calculations
  - Company data fetching
- **Tools Used**: 
  - Tavily Search API
  - Web scraping
  - Vector database (ChromaDB)
  - Yahoo Finance (yfinance)
  - Smart Calculator

**Code Location:** `src/agents/research_agent.py`

---

### 3.3 **IT Sector Agent** (`it_sector_agent.py`)
- **Role**: Specialized agent for IT/Technology sector analysis
- **Goal**: Provide expert insights on IT companies and trends
- **Capabilities**:
  - IT company analysis
  - Technology trend identification
  - Sector-specific metrics
  - Competitive analysis
- **Tools Used**: 
  - Sector-specific prompts
  - Specialized data sources

**Code Location:** `src/agents/it_sector_agent.py`

---

### 3.4 **Pharma Sector Agent** (`pharma_sector_agent.py`)
- **Role**: Specialized agent for pharmaceutical sector analysis
- **Goal**: Expert analysis of pharma companies and healthcare trends
- **Capabilities**:
  - Pharmaceutical company research
  - Drug pipeline analysis
  - Regulatory environment analysis
  - Healthcare market trends
- **Tools Used**: 
  - Sector-specific prompts
  - Healthcare data sources

**Code Location:** `src/agents/pharma_sector_agent.py`

---

### 3.5 **Query Classifier** (`query_classifier.py`)
- **Role**: Intelligent query classification and complexity assessment
- **Goal**: Determine optimal processing mode for each query
- **Capabilities**:
  - Complexity scoring (0-10 scale)
  - Mode classification (INSTANT/SIMPLE/COMPLEX/DEEP)
  - Confidence assessment
  - Step estimation
  - Auto-execute decision
- **Modes**:
  - **INSTANT** (0-2): Simple lookups, greetings
  - **SIMPLE** (3-4): Basic calculations, single data points
  - **COMPLEX** (5-7): Multi-step analysis, comparisons
  - **DEEP** (8-10): Comprehensive research, sector analysis

**Code Location:** `src/agents/query_classifier.py`

---

### 3.6 **Query Router** (`query_router.py`)
- **Role**: Route queries to appropriate specialized agents
- **Goal**: Ensure queries are handled by domain experts
- **Capabilities**:
  - Sector detection (IT, Pharma, Finance)
  - Company identification
  - Agent selection logic
  - Fallback to general research agent
- **Routing Logic**:
  ```
  IT Companies → IT Sector Agent
  Pharma Companies → Pharma Sector Agent
  General Queries → Research Agent
  ```

**Code Location:** `src/agents/query_router.py`

---

### 3.7 **Research Planner** (`research_planner.py`)
- **Role**: Generate detailed research plans for complex queries
- **Goal**: Break down complex research into manageable steps
- **Capabilities**:
  - Multi-step plan generation
  - Task decomposition
  - Resource identification
  - Timeline estimation
- **Output**: 
  - Structured research plan with 5-12 steps
  - Expected outputs for each step
  - Data sources for each step

**Code Location:** `src/agents/research_planner.py`

---

### 3.8 **Deep Research Executor** (`deep_research_executor.py`)
- **Role**: Execute complex multi-step research plans
- **Goal**: Conduct comprehensive deep research with progress tracking
- **Capabilities**:
  - Plan execution management
  - Step-by-step progress tracking
  - Multi-source data gathering
  - Synthesis and report generation
  - Error handling and recovery
- **Features**:
  - Real-time progress updates
  - Source tracking
  - Intermediate result storage
  - Final report compilation

**Code Location:** `src/agents/deep_research_executor.py`

---

### 3.9 **Orchestrator** (`orchestrator.py`)
- **Role**: Coordinate multiple agents for complex workflows
- **Goal**: Manage agent collaboration and task distribution
- **Capabilities**:
  - Multi-agent coordination
  - Task assignment
  - Result aggregation
  - Workflow management

**Code Location:** `src/agents/orchestrator.py`

---

## 4. AI Models Used

### Primary Language Model: **GPT-4o-mini**
```python
Model: "gpt-4o-mini"
Provider: OpenAI
Temperature: 0.1 (for precise, consistent outputs)
```

**Characteristics:**
- Cost-effective variant of GPT-4
- Optimized for agent-based workflows
- Balance between performance and cost
- Suitable for multi-agent systems
- **Cost**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens

**Usage in Project:**
- Agent reasoning and decision-making
- Natural language understanding
- Content generation
- Task planning and delegation
- Query classification

---

### Vision Model: **GPT-4o** (for PDF Charts)
```python
Model: "gpt-4o"
Provider: OpenAI
Use Case: Chart and image analysis in PDFs
```

**Usage:**
- Analyzing charts and graphs in PDF documents
- Extracting insights from financial visualizations
- Describing complex diagrams

**Code Location:** `src/utils/pdf_parser.py` (line ~350)

---

### Embedding Model: **all-MiniLM-L6-v2**
```python
Model: "sentence-transformers/all-MiniLM-L6-v2"
Provider: Hugging Face (local)
Embedding Dimension: 384
```

**Usage:**
- Document embedding for RAG
- Semantic search in vector database
- Similarity matching for context retrieval

**Code Location:** `src/data/vector_store.py`

---

## 5. APIs and Integrations

### 5.1 **OpenAI API**
**Purpose**: Primary language model and vision model access

**Models Used:**
- `gpt-4o-mini` - Main agent LLM
- `gpt-4o` - Vision model for chart analysis

**Endpoints Used:**
- `/v1/chat/completions` - For agent conversations and reasoning
- Vision API - For analyzing charts/images in PDFs

**Configuration:**
```python
api_key: OPENAI_API_KEY (from .env)
base_url: https://api.openai.com/v1
temperature: 0.1
max_tokens: 4000 (configurable)
```

**Usage Locations:**
- `src/core/llm_client.py` - Main LLM wrapper
- `src/utils/pdf_parser.py` - Vision API for charts
- All agent classes - For reasoning and responses

**Cost Estimate:**
- GPT-4o-mini: $0.15/1M input tokens, $0.60/1M output tokens
- GPT-4o (Vision): $2.50/1M input tokens, $10/1M output tokens

---

### 5.2 **Tavily Search API**
**Purpose**: Advanced AI-optimized web search

**Features:**
- Real-time web search
- Content extraction and summarization
- Source credibility ranking
- AI-optimized results
- Multi-query optimization

**Configuration:**
```python
api_key: TAVILY_API_KEY (from .env)
search_depth: "advanced"
max_results: 10
include_answer: True
include_raw_content: False
```

**Usage:**
- Research tasks
- Current events lookup
- Competitive analysis
- Trend identification
- Deep research mode

**Code Location:** `src/core/api_client.py` (TavilyClient class)

**API Details:**
- Endpoint: `https://api.tavily.com/search`
- Method: POST
- Response: JSON with results, answer, and sources

**Cost:**
- Free tier: 1,000 requests/month
- Paid: $50/month for 5,000 requests

---

### 5.3 **Yahoo Finance API (yfinance)**
**Purpose**: Financial data and stock market information

**Data Available:**
- Stock prices (real-time and historical)
- Company fundamentals
- Financial statements
- Market indices
- Trading volumes
- Analyst recommendations
- Company info (sector, industry, country)

**Configuration:**
```python
import yfinance as yf
# No API key required - free service
```

**Usage:**
- Stock price lookups
- Company data retrieval
- Market trend analysis
- Financial metrics calculation
- Historical data analysis

**Code Location:** `src/core/api_client.py` (get_company_info method)

**Example Data Retrieved:**
```python
{
    'company': 'Reliance Industries',
    'ticker': 'RELIANCE.NS',
    'sector': 'Energy',
    'country': 'India',
    'current_price': 2450.50,
    'currency': 'INR',
    'market_cap': 16500000000000,
    'pe_ratio': 28.5
}
```

**Advantages:**
- Free (no API key needed)
- Comprehensive data
- Global coverage
- No rate limits

---

### 5.4 **Web Scraping**

**Libraries Used:**
- **BeautifulSoup4**: HTML parsing
- **Requests**: HTTP requests
- **Selenium** (if needed): Dynamic content

**Capabilities:**
- Custom website data extraction
- Article content retrieval
- Structured data extraction
- Multiple source aggregation

**Code Location:** `src/core/api_client.py` (scrape_urls method)

**Implementation:**
```python
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract relevant content
    return extracted_data
```

**Usage:**
- Tavily search result enrichment
- Specific source data extraction
- Research content gathering

---

### 5.5 **ChromaDB** (Vector Database)

**Purpose**: Embedded vector database for RAG (Retrieval Augmented Generation)

**Features:**
- Document storage
- Semantic search
- Embedding generation
- Context retrieval
- PDF content indexing

**Configuration:**
```python
persist_directory: "./data/vector_db"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
collection_name: "financial_docs"
```

**Usage:**
- PDF content storage and retrieval
- Document chunking and embedding
- Semantic search for relevant context
- RAG pipeline for enhanced answers

**Code Location:** `src/data/vector_store.py`

**Workflow:**
```
PDF Upload → Text Extraction → Chunking (1500 chars) → 
Embedding (all-MiniLM-L6-v2) → ChromaDB Storage → 
Query → Semantic Search → Context Retrieval → Enhanced Answer
```

---

## 6. Tools and Utilities

### 6.1 **Enhanced PDF Parser** (`pdf_parser.py`)

**Purpose**: Multimodal PDF extraction (text, tables, charts)

**Libraries:**
- **PyMuPDF (fitz)**: Text and image extraction
- **pdfplumber**: Table detection and extraction
- **camelot-py**: Advanced table parsing
- **opencv-python**: Image processing
- **Pillow**: Image format conversion

**Capabilities:**
1. **Text Extraction**
   - Page-by-page text extraction
   - Layout preservation
   - Character encoding handling

2. **Table Extraction**
   - Automatic table detection
   - Markdown format conversion
   - Complex table handling
   - Fallback mechanisms (pdfplumber → camelot)

3. **Image Extraction**
   - Image identification and extraction
   - Base64 encoding
   - Format conversion (to PNG for Vision API)

4. **Chart Analysis**
   - Chart detection (images >200x200px)
   - Vision API integration (GPT-4o)
   - Chart description generation
   - "NOT_A_CHART" filtering

**Code Location:** `src/utils/pdf_parser.py`

**Usage Example:**
```python
from src.utils.pdf_parser import EnhancedPDFParser

parser = EnhancedPDFParser(use_vision_api=True)
result = parser.parse_pdf("annual-report.pdf")

print(f"Text: {len(result.text)} chars")
print(f"Tables: {len(result.tables)}")
print(f"Charts: {len(result.charts)}")
```

**Test Results** (from `test_enhanced_pdf.py`):
- Text extracted: 35,549 characters
- Tables extracted: 17 tables
- Images found: 44 images
- Charts analyzed: 30 charts
- Combined content: 53,244 characters (50% more than text-only)

---

### 6.2 **Smart Financial Calculator** (`src/utils/calculator.py`)

**Purpose**: Automated financial calculations

**Supported Formulas** (15+ calculations):

1. **P/E Ratio**: Price-to-Earnings Ratio
2. **ROE**: Return on Equity
3. **ROA**: Return on Assets
4. **Debt-to-Equity Ratio**
5. **Current Ratio**
6. **Quick Ratio**
7. **Gross Profit Margin**
8. **Net Profit Margin**
9. **Operating Profit Margin**
10. **EBITDA Margin**
11. **CAGR**: Compound Annual Growth Rate
12. **EPS**: Earnings Per Share
13. **Book Value Per Share**
14. **Dividend Yield**
15. **PEG Ratio**: Price/Earnings to Growth

**Auto-Detection:**
- Extracts numbers from natural language
- Identifies formula from context
- Performs calculation
- Provides interpretation

**Code Location:** `src/utils/calculator.py`

**Usage:**
```python
from src.utils.calculator import FinancialCalculator

calc = FinancialCalculator()
result = calc.calculate("P/E ratio if price is 2000 and EPS is 80")
# Output: "P/E Ratio = 25"
```

---

### 6.3 **Universal Company Resolver** (`universal_company_resolver.py`)

**Purpose**: Resolve company names to tickers and fetch data

**Capabilities:**
- Company name extraction from queries
- Ticker symbol resolution
- International company support
- Multiple company handling
- Ambiguity resolution

**Process:**
1. Extract company names using LLM
2. Resolve to ticker symbols
3. Validate tickers via yfinance
4. Fetch company data
5. Return structured information

**Code Location:** `src/utils/universal_company_resolver.py`

**Example:**
```python
Input: "Compare TCS and Infosys"

Output: {
    'resolved': True,
    'multiple': True,
    'companies': [
        {
            'company': 'Tata Consultancy Services',
            'ticker': 'TCS.NS',
            'sector': 'Technology',
            'current_price': 3450.20,
            'currency': 'INR'
        },
        {
            'company': 'Infosys Limited',
            'ticker': 'INFY.NS',
            'sector': 'Technology',
            'current_price': 1520.30,
            'currency': 'INR'
        }
    ]
}
```

---

### 6.4 **Document Processor** (`document_processor.py`)

**Purpose**: Chunk documents for vector storage

**Features:**
- Text chunking (1500 chars with 200 char overlap)
- Metadata preservation
- Batch processing
- Chunk limit management (500 chunks max)

**Code Location:** `src/data/document_processor.py`

**Chunking Strategy:**
```python
chunk_size: 1500 characters
chunk_overlap: 200 characters
max_chunks: 500 per document
```

---

### 6.5 **Context Manager** (`ConversationContextManager`)

**Purpose**: Manage conversation history and context

**Features:**
- Multi-turn conversation tracking
- Context resolution with AI
- Session management
- History pruning
- Company/entity tracking

**Code Location:** `src/agents/query_classifier.py` (lines 338+)

**Context Resolution:**
- Detects pronouns ("it", "they", "this")
- Replaces with actual company names
- Maintains conversation flow
- Handles follow-up questions

---

## 7. API Keys Required

### Environment Variables Setup

Create a `.env` file in project root:

```bash
# Required API Keys
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional API Keys (for future enhancements)
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
SERPER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
PORT=8000
HOST=127.0.0.1
```

---

### API Key Acquisition Guide:

#### **1. OpenAI API Key** ⭐ REQUIRED
**Steps:**
1. Visit: https://platform.openai.com/api-keys
2. Create account or login
3. Click "Create new secret key"
4. Name your key (e.g., "financial-research-agent")
5. Copy key immediately (won't be shown again)
6. Add billing information in Settings → Billing
7. Add credits (minimum $5 recommended)

**Cost Estimate:**
- GPT-4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output tokens
- Average query: ~$0.001 - $0.005
- 1000 queries/month: ~$3-5/month

**Usage Limits:**
- Free tier: $5 credit (expires after 3 months)
- Paid tier: Set your own limits

---

#### **2. Tavily API Key** ⭐ REQUIRED
**Steps:**
1. Visit: https://tavily.com/
2. Sign up with email
3. Verify email
4. Access dashboard
5. Copy API key from dashboard

**Pricing:**
- **Free Tier**: 1,000 requests/month
- **Basic**: $50/month for 5,000 requests
- **Pro**: $150/month for 20,000 requests

**Usage in Project:**
- Deep research mode
- Web search for complex queries
- Real-time information retrieval

---

#### **3. Yahoo Finance** ✅ NO KEY REQUIRED
**Setup:**
```bash
pip install yfinance
```

**Free Features:**
- Stock prices
- Company fundamentals
- Historical data
- No rate limits
- No authentication

---

#### **4. Hugging Face** ⚠️ OPTIONAL
**Purpose:** Access to additional models or datasets

**Steps:**
1. Visit: https://huggingface.co/
2. Create account
3. Settings → Access Tokens
4. Generate new token
5. Select "Read" permission

**Cost:** Free (for most models)

**Use Cases:**
- Alternative embedding models
- Specialized AI models
- Dataset access

---

## 8. Agent Workflows

### 8.1 **Query Processing Workflow**

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Query Classifier    │ ◄── Complexity analysis (0-10 scale)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Mode Decision       │
├─────────────────────┤
│ INSTANT (0-2)       │ → Direct response
│ SIMPLE (3-4)        │ → Single agent + tools
│ COMPLEX (5-7)       │ → Multi-step analysis
│ DEEP (8-10)         │ → Research plan generation
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Query Router        │ ◄── Route to specialized agent
└──────┬──────────────┘
       │
       ├─────► IT Sector Agent (TCS, Infosys, etc.)
       ├─────► Pharma Agent (Sun Pharma, Dr. Reddy's, etc.)
       └─────► Research Agent (General queries)
       
       │
       ▼
┌─────────────────────┐
│ Tool Execution      │
├─────────────────────┤
│ • Tavily Search     │
│ • Yahoo Finance     │
│ • Web Scraping      │
│ • Calculator        │
│ • Vector Store RAG  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Response Generation │ ◄── LLM synthesis
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ User Response       │
└─────────────────────┘
```

---

### 8.2 **PDF Processing Workflow**

```
┌──────────────┐
│ PDF Upload   │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ EnhancedPDFParser    │
└──────┬───────────────┘
       │
       ├─────► PyMuPDF: Text extraction (page-by-page)
       │
       ├─────► pdfplumber: Table detection
       │        │
       │        └──► camelot: Complex tables (fallback)
       │
       └─────► Image extraction + Chart detection
                │
                └──► GPT-4o Vision: Chart analysis
                
       │
       ▼
┌──────────────────────┐
│ Combined Content     │
├──────────────────────┤
│ • Text: ~35K chars   │
│ • Tables: 17 (MD)    │
│ • Charts: 30 (desc)  │
│ Total: ~53K chars    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Document Processor   │ ◄── Chunking (1500 chars)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ ChromaDB Storage     │ ◄── Vector embeddings
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Ready for Queries    │
└──────────────────────┘
```

---

### 8.3 **Deep Research Workflow**

```
┌─────────────────┐
│ Complex Query   │ (e.g., "Pharma sector analysis")
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Classifier          │ → Complexity: 8-10/10
│ Result: DEEP mode   │ → Auto-execute: False
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Research Planner    │ ◄── Generate 5-12 step plan
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Show Plan to User   │
│ ┌─────────────────┐ │
│ │ Step 1: Market  │ │
│ │ Step 2: Players │ │
│ │ Step 3: Trends  │ │
│ │ ...             │ │
│ └─────────────────┘ │
│ [Confirm Execution] │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Deep Executor       │
└────────┬────────────┘
         │
         ├──► Step 1: Tavily Search → Market Size
         ├──► Step 2: Company Data → Key Players
         ├──► Step 3: Tavily Search → Growth Drivers
         ├──► Step 4: Web Scraping → Challenges
         ├──► Step 5: Synthesis → Final Report
         │
         ▼
┌─────────────────────┐
│ Comprehensive       │
│ Research Report     │
├─────────────────────┤
│ • 2000+ words       │
│ • 10+ sources       │
│ • Data tables       │
│ • Executive summary │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ User Receives       │
│ Detailed Report     │
└─────────────────────┘
```

---

### 8.4 **Financial Calculation Workflow**

```
┌─────────────────────┐
│ Query with Numbers  │ (e.g., "P/E if price 2000, EPS 80")
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Number Extractor    │ ◄── Extract: [2000, 80]
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Formula Detector    │ ◄── Identify: "P/E Ratio"
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Calculator          │
│ P/E = 2000 / 80     │
│ P/E = 25            │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Formatted Response  │
├─────────────────────┤
│ P/E Ratio = 25      │
│                     │
│ Formula:            │
│ P/E = Price / EPS   │
│                     │
│ Interpretation:     │
│ [LLM-generated]     │
└─────────────────────┘
```

---

## 9. Setup and Configuration

### 9.1 **Installation Steps**

```bash
# 1. Navigate to project directory
cd C:\Users\Jeet\financial_research_agent\financial_research_agent

# 2. Create virtual environment (if not exists)
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create .env file
# Copy .env.example to .env and add your API keys

# 6. Run the application
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

---

### 9.2 **Dependencies** (`requirements.txt`)

```txt
# Web Framework
fastapi==0.109.2
uvicorn[standard]==0.27.1
python-multipart==0.0.9

# AI & LLM
openai==1.12.0
langchain==0.1.9
langchain-community==0.0.20

# PDF Processing
PyMuPDF==1.23.8
pdfplumber==0.10.3
camelot-py==0.11.0
opencv-python==4.9.0.80
Pillow==10.2.0

# Vector Database & Embeddings
chromadb==0.4.22
sentence-transformers==2.3.1

# Search & Scraping
tavily-python==0.3.0
beautifulsoup4==4.12.3
requests==2.31.0
lxml==5.1.0

# Financial Data
yfinance==0.2.36

# Data Processing
pandas==2.2.0
numpy==1.26.4

# Utilities
python-dotenv==1.0.1
pydantic==2.6.1
```

---

### 9.3 **Project Structure**

```
financial_research_agent/
│
├── src/
│   ├── __init__.py
│   ├── api.py                    # FastAPI main application
│   │
│   ├── agents/                   # All agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Base agent class
│   │   ├── research_agent.py     # General research agent
│   │   ├── it_sector_agent.py    # IT sector specialist
│   │   ├── pharma_sector_agent.py # Pharma specialist
│   │   ├── query_classifier.py   # Query classification
│   │   ├── query_router.py       # Agent routing
│   │   ├── research_planner.py   # Deep research planning
│   │   ├── deep_research_executor.py # Plan execution
│   │   └── orchestrator.py       # Multi-agent coordination
│   │
│   ├── core/                     # Core functionality
│   │   ├── __init__.py
│   │   ├── llm_client.py         # OpenAI LLM wrapper
│   │   ├── api_client.py         # Tavily, yfinance, scraping
│   │   ├── research_engine.py    # Research orchestration
│   │   └── report_generator.py   # Report creation
│   │
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   ├── vector_store.py       # ChromaDB operations
│   │   └── document_processor.py # Text chunking
│   │
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── pdf_parser.py         # Enhanced PDF processing
│   │   ├── calculator.py         # Financial calculations
│   │   └── universal_company_resolver.py # Company resolution
│   │
│   └── config/                   # Configuration
│       ├── __init__.py
│       ├── settings.py           # App settings
│       ├── prompts.py            # LLM prompts
│       └── agent_configs.py      # Agent configurations
│
├── static/                       # Frontend
│   ├── index.html
│   ├── css/
│   └── js/
│
├── data/                         # Data storage
│   ├── vector_db/                # ChromaDB storage
│   ├── cache/
│   └── processed/
│
├── logs/                         # Application logs
│   └── agent.log
│
├── uploads/                      # User uploaded files
│
├── .env                          # Environment variables (API keys)
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── COMPREHENSIVE_TEST_CASES.md   # Test cases
```

---

### 9.4 **Configuration Files**

#### **settings.py** (`src/config/settings.py`)

Key configurations:
```python
# LLM Settings
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.1
MAX_TOKENS = 4000

# Vector Store
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_CHUNKS = 500

# API Settings
TAVILY_MAX_RESULTS = 10
SEARCH_DEPTH = "advanced"

# PDF Processing
USE_VISION_API = True
VISION_MODEL = "gpt-4o"
CHART_MIN_SIZE = 200  # pixels
```

---

#### **prompts.py** (`src/config/prompts.py`)

System prompts for agents:
```python
RESEARCH_AGENT_PROMPT = """
You are a financial research expert...
"""

IT_SECTOR_PROMPT = """
You specialize in IT/Technology sector analysis...
"""

PHARMA_SECTOR_PROMPT = """
You are an expert in pharmaceutical industry analysis...
"""
```

---

## 10. How Each Component Works

### 10.1 **Query Classifier - Detailed Logic**

**File:** `src/agents/query_classifier.py`

**Classification Process:**

```python
def classify_query(query: str) -> ClassificationResult:
    # 1. Calculate complexity score (0-10)
    score = 0
    
    # Greeting check
    if is_greeting(query):
        return INSTANT (score=0)
    
    # Word count factor
    words = len(query.split())
    if words > 30: score += 2
    
    # Keyword detection
    deep_keywords = ['analysis', 'research', 'compare', 'sector', 'outlook']
    calc_keywords = ['calculate', 'compute', 'ratio', 'margin']
    
    if any(kw in query for kw in deep_keywords):
        score += 3
    
    # Company count
    if multiple_companies_detected:
        score += 2
    
    # Time range detection
    if 'last 3 years' or 'historical':
        score += 2
    
    # 2. Map score to mode
    if score <= 2: return INSTANT
    elif score <= 4: return SIMPLE
    elif score <= 7: return COMPLEX
    else: return DEEP
```

**Confidence Calculation:**
- INSTANT: 0.70 (high certainty for simple queries)
- SIMPLE: 0.65
- COMPLEX: 0.60
- DEEP: 0.55 (lower certainty, but comprehensive approach)

---

### 10.2 **Universal Company Resolver - Process**

**File:** `src/utils/universal_company_resolver.py`

**Resolution Steps:**

```python
def resolve_company(query: str) -> dict:
    # Step 1: Extract company names using LLM
    prompt = f"Extract company names from: {query}"
    companies = llm.extract_entities(prompt)
    # Output: ['TCS', 'Infosys']
    
    # Step 2: Map to ticker symbols
    tickers = []
    for company in companies:
        # Try direct mapping
        ticker = lookup_ticker(company)
        
        # Try fuzzy matching
        if not ticker:
            ticker = fuzzy_search(company)
        
        tickers.append(ticker)
    # Output: ['TCS.NS', 'INFY.NS']
    
    # Step 3: Validate and fetch data
    validated = []
    for ticker in tickers:
        data = yfinance.Ticker(ticker)
        if data.info:
            validated.append({
                'company': data.info['longName'],
                'ticker': ticker,
                'sector': data.info['sector'],
                'current_price': data.info['currentPrice']
            })
    
    return {
        'resolved': True,
        'multiple': len(validated) > 1,
        'companies': validated
    }
```

---

### 10.3 **Enhanced PDF Parser - Technical Details**

**File:** `src/utils/pdf_parser.py`

**Parsing Pipeline:**

```python
class EnhancedPDFParser:
    def parse_pdf(self, filepath: str) -> PDFContent:
        # 1. Text Extraction (PyMuPDF)
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        
        # 2. Table Extraction (pdfplumber + camelot)
        tables = []
        
        # Try pdfplumber first (faster)
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                tables.extend(page_tables)
        
        # Fallback to camelot for complex tables
        if len(tables) < expected:
            camelot_tables = camelot.read_pdf(
                filepath, 
                flavor='lattice'
            )
            tables.extend(camelot_tables)
        
        # 3. Image & Chart Extraction
        images = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img in image_list:
                # Extract image
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                # Convert to PNG if needed
                if pix.n not in [1, 3, 4]:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                # Save as base64
                img_data = pix.tobytes("png")
                img_base64 = base64.b64encode(img_data)
                
                images.append({
                    'page': page_num,
                    'data': img_base64,
                    'width': pix.width,
                    'height': pix.height
                })
        
        # 4. Chart Analysis (Vision AI)
        charts = []
        for img in images:
            # Filter potential charts (size threshold)
            if img['width'] > 200 and img['height'] > 200:
                description = self._analyze_chart_with_vision(
                    img['data']
                )
                
                if "NOT_A_CHART" not in description:
                    charts.append({
                        'page': img['page'],
                        'description': description
                    })
        
        # 5. Combine all content
        combined = f"{text}\n\n"
        
        for i, table in enumerate(tables):
            combined += f"Table {i+1}:\n{table_to_markdown(table)}\n\n"
        
        for i, chart in enumerate(charts):
            combined += f"Chart {i+1} (Page {chart['page']}):\n"
            combined += f"{chart['description']}\n\n"
        
        return PDFContent(
            text=text,
            tables=tables,
            images=images,
            charts=charts,
            combined_text=combined
        )
```

---

### 10.4 **Deep Research Executor - Step-by-Step**

**File:** `src/agents/deep_research_executor.py`

**Execution Flow:**

```python
async def execute_plan(plan: ResearchPlan) -> ResearchResult:
    results = []
    
    for step in plan.steps:
        print(f"🔄 Step {step.number}/{plan.total_steps}: {step.title}")
        
        # Step 1: Determine tools needed
        if step.requires_search:
            search_results = await tavily.search(step.query)
            results.append(search_results)
        
        # Step 2: If company data needed
        if step.requires_company_data:
            company_data = yfinance.get_data(step.companies)
            results.append(company_data)
        
        # Step 3: If web scraping needed
        if step.requires_scraping:
            scraped_data = await scrape_urls(step.urls)
            results.append(scraped_data)
        
        # Step 4: Synthesize step results
        step_summary = llm.synthesize(results)
        
        print(f"✅ Step {step.number} completed")
    
    # Final synthesis
    final_report = llm.generate_report(
        query=plan.query,
        all_results=results,
        structure=plan.expected_outputs
    )
    
    return ResearchResult(
        answer=final_report,
        sources=collect_all_sources(results),
        iteration_count=len(plan.steps)
    )
```

---

## 11. Performance Metrics

### Current Performance (from logs):

**Query Processing Times:**
- INSTANT queries: 2-5 seconds
- SIMPLE queries: 5-10 seconds
- COMPLEX queries: 15-30 seconds
- DEEP research: 30-120 seconds

**PDF Processing:**
- Upload: < 5 seconds
- Text extraction: 5-10 seconds
- Table extraction: 10-20 seconds
- Chart analysis (30 charts): 60-90 seconds
- Total (35-page PDF): ~2 minutes

**Vector Store:**
- Document chunking: 1-2 seconds
- Embedding generation (500 chunks): 30-40 seconds
- Semantic search: < 1 second

---

## 12. Cost Analysis

### Per 1000 Queries (Mixed):

**OpenAI Costs:**
- 600 INSTANT (GPT-4o-mini, ~500 tokens): $0.05
- 300 SIMPLE (GPT-4o-mini, ~1000 tokens): $0.09
- 80 COMPLEX (GPT-4o-mini, ~2000 tokens): $0.10
- 20 DEEP (GPT-4o-mini, ~5000 tokens): $0.15
- **Total OpenAI**: ~$0.39

**PDF Processing (100 PDFs with charts):**
- Vision API calls (30 charts × 100 PDFs): $7.50
- **Total with PDFs**: ~$7.89

**Tavily Search:**
- 200 deep research queries × 10 searches: 2000 requests
- Free tier: 1000 free/month
- Paid: $50/month for 5000
- **Cost for 1000 extra**: ~$10

**Total Monthly (1000 queries + 100 PDFs):**
- OpenAI: $0.39
- OpenAI Vision: $7.50
- Tavily: $10
- **Total: ~$18/month**

---

## 13. Troubleshooting

### Common Issues:

**1. "No module named 'src'"**
```bash
# Ensure you're in the correct directory
cd C:\Users\Jeet\financial_research_agent\financial_research_agent

# Run with python -m
python -m uvicorn src.api:app --reload
```

**2. "OpenAI API key not found"**
```bash
# Check .env file exists and has:
OPENAI_API_KEY=sk-...

# Restart server after adding key
```

**3. "Vector store empty"**
```bash
# Upload a PDF first, then query
# Or check if vector_db directory exists
```

**4. "Tavily search failed"**
```bash
# Check Tavily API key
# Check monthly quota (1000 free)
# Fallback to web scraping if needed
```

---

## 14. Future Enhancements

**Planned Features:**
1. ✅ Enhanced PDF processing (DONE)
2. ✅ Deep research mode (DONE)
3. ✅ Multi-modal analysis (DONE)
4. 🔲 Excel/CSV file support
5. 🔲 Chart generation (matplotlib)
6. 🔲 Multi-language support
7. 🔲 Voice input/output
8. 🔲 Real-time stock alerts
9. 🔲 Portfolio tracking
10. 🔲 Automated report scheduling

---

## Conclusion

This financial research agent system leverages:

- **9 Specialized Agents** for different tasks
- **GPT-4o-mini** as primary reasoning engine
- **GPT-4o Vision** for chart analysis
- **Multiple APIs** (OpenAI, Tavily, Yahoo Finance)
- **Advanced Tools** (PDF parser, calculator, company resolver)
- **RAG Pipeline** with ChromaDB
- **Custom Architecture** (not CrewAI)

The system is designed for:
- ✅ Scalability
- ✅ Modularity
- ✅ Cost efficiency
- ✅ High accuracy
- ✅ Comprehensive research capabilities

---

## Quick Reference

### Start Server
```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

### Access UI
```
http://127.0.0.1:8000
```

### Test API
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is TCS stock price?", "session_id": "test123"}'
```

### Check Logs
```bash
tail -f logs/agent.log
```

---

**Document Version:** 1.0  
**Last Updated:** February 5, 2026  
**Primary Model:** GPT-4o-mini  
**Framework:** Custom Multi-Agent System  
**Total Lines of Code:** 8,000+
