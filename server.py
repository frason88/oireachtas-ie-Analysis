import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

def start_server():
    """Start a simple HTTP server to serve the dashboard"""
    PORT = 8000
    
    # Change to the directory containing the dashboard files
    os.chdir(Path(__file__).parent)
    
    # Create HTTP server
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 Dashboard server started at http://localhost:{PORT}")
        print(f"📊 Open your browser and navigate to: http://localhost:{PORT}")
        print("Press Ctrl+C to stop the server")
        
        # Try to open the browser automatically
        try:
            webbrowser.open(f'http://localhost:{PORT}')
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
            httpd.shutdown()

if __name__ == "__main__":
    start_server() 