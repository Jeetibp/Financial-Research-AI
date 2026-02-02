"""
Simple Research Agent Test
"""
print("="*70)
print("STARTING SIMPLE TEST...")
print("="*70)

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")

import asyncio
from src.agents.research_agent import ResearchAgent
from src.core.report_generator import ReportGenerator

print("\n✅ Imports successful!")

async def main():
    """Run simple test"""
    
    print("\n" + "="*70)
    print("INITIALIZING AGENT...")
    print("="*70)
    
    try:
        agent = ResearchAgent()
        print("✅ Agent initialized!")
        
        report_gen = ReportGenerator()
        print("✅ Report generator initialized!")
        
        # Simple query
        query = "What is Microsoft's current stock price?"
        
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")
        
        print("\n⏳ Researching... (30-60 seconds)")
        
        result = await agent.research(query, use_web=True)
        
        print("\n✅ Research complete!")
        print(f"\nAnswer Preview: {result.answer[:200]}...")
        
        # Save report
        html_path = report_gen.save_report(result, format="html")
        
        print(f"\n{'='*70}")
        print("REPORT SAVED!")
        print(f"{'='*70}")
        print(f"📁 {html_path}")
        
        # Open in browser
        import webbrowser
        abs_path = Path(html_path).absolute()
        file_url = abs_path.as_uri()
        
        print(f"\n🌐 Opening: {file_url}")
        
        try:
            webbrowser.open(file_url)
            print("✅ Opened in browser!")
        except:
            print("⚠️ Could not auto-open. Open manually.")
        
        print(f"\n{'='*70}")
        print("✅ TEST COMPLETE!")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n⏳ Starting async event loop...")
    asyncio.run(main())
    print("\n🎉 Done!")
