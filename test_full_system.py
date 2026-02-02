"""
Report Generator
Generate HTML and text reports from research results
"""

from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from src.agents.research_agent import ResearchResult
from src.utils.logger import get_logger

logger = get_logger("report_generator")

class ReportGenerator:
    """Generate formatted research reports"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Report generator initialized: {self.output_dir}")
    
    def generate_html_report(self, result: ResearchResult) -> str:
        """Generate HTML report from research result"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build sources HTML
        sources_html = ""
        for i, source in enumerate(result.sources, 1):
            sources_html += f"""
            <div class="source">
                <div class="source-title">{i}. {source['title']}</div>
                <div class="source-url">
                    <a href="{source['url']}" target="_blank" rel="noopener noreferrer">
                        🔗 {source['url']}
                    </a>
                </div>
                <p class="snippet">{source['snippet']}</p>
            </div>
            """
        
        # Build context HTML
        context_html = ""
        for i, chunk in enumerate(result.context_used[:3], 1):
            context_html += f"""
            <div class="context-chunk">
                <strong>Context {i}:</strong><br>
                <p>{chunk[:300]}...</p>
            </div>
            """
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report - {result.query}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .timestamp {{
            opacity: 0.9;
            font-size: 14px;
            margin-top: 10px;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
            margin-bottom: 20px;
        }}
        .query {{
            font-size: 20px;
            color: #333;
            font-weight: 500;
            margin-bottom: 15px;
        }}
        .answer {{
            color: #444;
            white-space: pre-wrap;
            line-height: 1.8;
        }}
        .source {{
            background: #f8f9fa;
            padding: 20px;
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
            border-radius: 4px;
            transition: transform 0.2s;
        }}
        .source:hover {{
            transform: translateX(5px);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        }}
        .source-title {{
            font-weight: 600;
            color: #333;
            margin-bottom: 12px;
            font-size: 16px;
        }}
        .source-url {{
            margin: 12px 0;
        }}
        .source a {{
            color: #667eea;
            text-decoration: none;
            word-break: break-all;
            font-size: 14px;
            padding: 8px 12px;
            background: #f0f4ff;
            border-radius: 6px;
            display: inline-block;
            transition: all 0.3s;
            border: 1px solid transparent;
        }}
        .source a:hover {{
            background: #667eea;
            color: white;
            text-decoration: none;
            border: 1px solid #667eea;
            transform: translateY(-2px);
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
        }}
        .snippet {{
            color: #666;
            font-size: 14px;
            margin-top: 12px;
            line-height: 1.6;
        }}
        .context-chunk {{
            background: #fff8e1;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 6px;
            font-size: 14px;
            border-left: 3px solid #ffd54f;
        }}
        .context-chunk strong {{
            color: #f57c00;
            display: block;
            margin-bottom: 8px;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 25px;
            border-radius: 8px;
            flex: 1;
            min-width: 150px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .stat-box:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
        }}
        .stat-number {{
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 8px;
        }}
        .stat-label {{
            font-size: 13px;
            opacity: 0.95;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        @media print {{
            body {{
                max-width: 100%;
                margin: 0;
                padding: 20px;
            }}
            .source:hover, .stat-box:hover {{
                transform: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Financial Research Report</h1>
        <div class="timestamp">Generated: {timestamp}</div>
    </div>
    
    <div class="section">
        <h2>📋 Research Query</h2>
        <div class="query">{result.query}</div>
    </div>
    
    <div class="section">
        <h2>💡 Analysis & Findings</h2>
        <div class="answer">{result.answer}</div>
    </div>
    
    <div class="section">
        <h2>📚 Sources ({len(result.sources)})</h2>
        {sources_html if sources_html else "<p>No sources available</p>"}
    </div>
    
    <div class="section">
        <h2>📊 Research Statistics</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{len(result.sources)}</div>
                <div class="stat-label">Sources Used</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{len(result.context_used)}</div>
                <div class="stat-label">Context Chunks</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{len(result.answer.split())}</div>
                <div class="stat-label">Words in Answer</div>
            </div>
        </div>
    </div>
    
    {f'''<div class="section">
        <h2>🔍 Retrieved Context</h2>
        {context_html}
    </div>''' if context_html else ''}
    
</body>
</html>
        """
        
        return html_template
    
    def save_report(self, result: ResearchResult, format: str = "html") -> str:
        """Save report to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c for c in result.query[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_query = safe_query.replace(' ', '_')
        
        if format == "html":
            filename = f"report_{safe_query}_{timestamp}.html"
            filepath = self.output_dir / filename
            
            html_content = self.generate_html_report(result)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved: {filepath}")
            return str(filepath)
        
        elif format == "txt":
            filename = f"report_{safe_query}_{timestamp}.txt"
            filepath = self.output_dir / filename
            
            txt_content = self._generate_text_report(result)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            
            logger.info(f"Text report saved: {filepath}")
            return str(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_text_report(self, result: ResearchResult) -> str:
        """Generate plain text report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
{'='*70}
FINANCIAL RESEARCH REPORT
{'='*70}
Generated: {timestamp}

QUERY:
{result.query}

{'='*70}
ANALYSIS & FINDINGS:
{'='*70}

{result.answer}

{'='*70}
SOURCES ({len(result.sources)}):
{'='*70}

"""
        
        for i, source in enumerate(result.sources, 1):
            report += f"{i}. {source['title']}\n"
            report += f"   URL: {source['url']}\n"
            report += f"   {source['snippet']}\n\n"
        
        report += f"{'='*70}\n"
        report += f"STATISTICS:\n"
        report += f"{'='*70}\n"
        report += f"Sources Used: {len(result.sources)}\n"
        report += f"Context Chunks: {len(result.context_used)}\n"
        report += f"Words in Answer: {len(result.answer.split())}\n"
        
        return report
