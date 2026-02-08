"""
Enhanced PDF Parser - Extract text, tables, charts, and images from PDFs
Supports multimodal extraction with GPT-4 Vision for chart/graph analysis
"""

import fitz  # PyMuPDF
import pdfplumber
import camelot
import base64
import io
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from PIL import Image
import openai
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger("pdf_parser")


@dataclass
class PDFContent:
    """Structured container for all PDF content"""
    text: str
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_combined_text(self) -> str:
        """Combine all content into a single text representation"""
        combined = [self.text]
        
        # Add table content
        if self.tables:
            combined.append("\n\n=== TABLES ===\n")
            for i, table in enumerate(self.tables, 1):
                combined.append(f"\n--- Table {i} (Page {table['page']}) ---\n")
                combined.append(table['markdown'])
        
        # Add chart/graph descriptions
        if self.charts:
            combined.append("\n\n=== CHARTS & GRAPHS ===\n")
            for i, chart in enumerate(self.charts, 1):
                combined.append(f"\n--- Chart {i} (Page {chart['page']}) ---\n")
                combined.append(chart['description'])
        
        # Add image descriptions
        if self.images:
            combined.append("\n\n=== IMAGES ===\n")
            for i, img in enumerate(self.images, 1):
                if img.get('description'):
                    combined.append(f"\n--- Image {i} (Page {img['page']}) ---\n")
                    combined.append(img['description'])
        
        return "\n".join(combined)


class EnhancedPDFParser:
    """
    Advanced PDF parser that extracts:
    - Text content (with layout preservation)
    - Tables (structured data)
    - Images (charts, graphs, diagrams)
    - Uses GPT-4 Vision for chart/graph interpretation
    """
    
    def __init__(self, use_vision_api: bool = True, openai_api_key: Optional[str] = None):
        """
        Initialize the PDF parser
        
        Args:
            use_vision_api: Enable GPT-4 Vision for chart/graph analysis
            openai_api_key: OpenAI API key (if not set in environment)
        """
        self.use_vision_api = use_vision_api
        
        if use_vision_api:
            if openai_api_key:
                openai.api_key = openai_api_key
            elif not os.getenv("OPENAI_API_KEY"):
                logger.warning("No OpenAI API key found. Vision features will be disabled.")
                self.use_vision_api = False
    
    def parse_pdf(self, pdf_path: str, analyze_charts: bool = True) -> PDFContent:
        """
        Extract all content from PDF
        
        Args:
            pdf_path: Path to PDF file
            analyze_charts: Use GPT-4 Vision to analyze charts/graphs
            
        Returns:
            PDFContent object with all extracted data
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        content = PDFContent(
            text="",
            metadata={
                "filename": pdf_path.name,
                "filepath": str(pdf_path.absolute())
            }
        )
        
        try:
            # Extract text and images
            text_parts, images = self._extract_with_pymupdf(str(pdf_path))
            content.text = "\n\n".join(text_parts)
            content.images = images
            
            # Extract tables
            content.tables = self._extract_tables(str(pdf_path))
            
            # Analyze charts/graphs with Vision API
            if analyze_charts and self.use_vision_api:
                content.charts = self._analyze_charts(images)
            
            logger.info(
                f"Extracted from {pdf_path.name}: "
                f"{len(content.text)} chars, "
                f"{len(content.tables)} tables, "
                f"{len(content.images)} images, "
                f"{len(content.charts)} charts"
            )
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path.name}: {e}")
            raise
        
        return content
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Tuple[List[str], List[Dict]]:
        """Extract text and images using PyMuPDF"""
        doc = fitz.open(pdf_path)
        text_parts = []
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text with layout
            text = page.get_text("text")
            if text.strip():
                text_parts.append(f"=== Page {page_num + 1} ===\n{text}")
            
            # Extract images
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to base64 for storage and API calls
                    image_base64 = base64.b64encode(image_bytes).decode()
                    
                    # Check if image looks like a chart/graph (by size)
                    # Charts are typically larger than icons
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    width, height = img_obj.size
                    
                    # Convert to PNG if not already (OpenAI requires png/jpeg/gif/webp)
                    if base_image['ext'] not in ['png', 'jpeg', 'jpg', 'gif', 'webp']:
                        buffer = io.BytesIO()
                        img_obj.save(buffer, format='PNG')
                        image_bytes = buffer.getvalue()
                        image_base64 = base64.b64encode(image_bytes).decode()
                        ext = 'png'
                    else:
                        ext = base_image['ext']
                    
                    is_likely_chart = (width > 200 and height > 200)
                    
                    images.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'data': image_base64,
                        'ext': ext,
                        'width': width,
                        'height': height,
                        'is_likely_chart': is_likely_chart
                    })
                    
                except Exception as e:
                    logger.warning(f"Could not extract image {img_index} from page {page_num + 1}: {e}")
        
        doc.close()
        return text_parts, images
    
    def _extract_tables(self, pdf_path: str) -> List[Dict]:
        """Extract tables using pdfplumber and camelot"""
        tables = []
        
        # Try pdfplumber first (better for simple tables)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    
                    for table_index, table_data in enumerate(page_tables):
                        if table_data and len(table_data) > 1:  # At least header + 1 row
                            markdown = self._table_to_markdown(table_data)
                            tables.append({
                                'page': page_num,
                                'index': table_index,
                                'data': table_data,
                                'markdown': markdown,
                                'rows': len(table_data),
                                'cols': len(table_data[0]) if table_data else 0,
                                'method': 'pdfplumber'
                            })
        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {e}")
        
        # Try camelot for complex tables (if pdfplumber found none)
        if not tables:
            try:
                camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
                
                for i, table in enumerate(camelot_tables):
                    df = table.df
                    markdown = df.to_markdown(index=False)
                    
                    tables.append({
                        'page': table.page,
                        'index': i,
                        'data': df.values.tolist(),
                        'markdown': markdown,
                        'rows': len(df),
                        'cols': len(df.columns),
                        'method': 'camelot',
                        'accuracy': table.accuracy
                    })
            except Exception as e:
                logger.warning(f"Camelot table extraction failed: {e}")
        
        logger.info(f"Extracted {len(tables)} tables")
        return tables
    
    def _table_to_markdown(self, table_data: List[List[str]]) -> str:
        """Convert table data to markdown format"""
        if not table_data:
            return ""
        
        lines = []
        
        # Header
        header = " | ".join(str(cell or "") for cell in table_data[0])
        lines.append(f"| {header} |")
        
        # Separator
        sep = " | ".join("---" for _ in table_data[0])
        lines.append(f"| {sep} |")
        
        # Data rows
        for row in table_data[1:]:
            row_text = " | ".join(str(cell or "") for cell in row)
            lines.append(f"| {row_text} |")
        
        return "\n".join(lines)
    
    def _analyze_charts(self, images: List[Dict]) -> List[Dict]:
        """Analyze charts and graphs using GPT-4 Vision"""
        charts = []
        
        for img in images:
            # Only analyze images that look like charts/graphs
            if not img.get('is_likely_chart', False):
                continue
            
            try:
                description = self._describe_image_with_vision(img['data'])
                
                if description:
                    charts.append({
                        'page': img['page'],
                        'index': img['index'],
                        'description': description,
                        'width': img['width'],
                        'height': img['height']
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to analyze chart on page {img['page']}: {e}")
        
        return charts
    
    def _describe_image_with_vision(self, image_base64: str) -> Optional[str]:
        """Use GPT-4 Vision to describe chart/graph"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",  # Updated model (gpt-4-vision-preview is deprecated)
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this chart/graph/diagram from a financial document. "
                                    "If this is not a chart/graph (e.g., photo, logo, decorative image), "
                                    "respond with just 'NOT_A_CHART'. Otherwise provide:\n"
                                    "1. Type of visualization (bar chart, line graph, pie chart, table, etc.)\n"
                                    "2. What data it shows (variables, metrics)\n"
                                    "3. Key insights and trends\n"
                                    "4. Specific numbers/values if visible\n"
                                    "Be concise but comprehensive."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            # Don't save if it's not actually a chart
            if "NOT_A_CHART" in content or "not a chart" in content.lower():
                return None
            
            return content
            
        except Exception as e:
            logger.error(f"GPT-4 Vision API error: {e}")
            return None
    
    def extract_page_as_image(self, pdf_path: str, page_num: int, dpi: int = 300) -> Optional[str]:
        """
        Render entire page as image and analyze with Vision API
        Useful as fallback for complex layouts
        
        Args:
            pdf_path: Path to PDF
            page_num: Page number (0-indexed)
            dpi: Resolution for rendering
            
        Returns:
            Base64 encoded image
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # Render page as image
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes()
            img_base64 = base64.b64encode(img_bytes).decode()
            
            doc.close()
            return img_base64
            
        except Exception as e:
            logger.error(f"Error rendering page {page_num}: {e}")
            return None
