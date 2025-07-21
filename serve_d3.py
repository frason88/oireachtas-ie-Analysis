import http.server
import socketserver
import webbrowser
import os
import socket

def find_free_port():
    """Find a free port to use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_server():
    """Start a simple HTTP server to serve the D3.js visualization"""
    port = find_free_port()
    
    with socketserver.TCPServer(("", port), MyHTTPRequestHandler) as httpd:
        print(f"Server started at http://localhost:{port}")
        print("Opening D3.js visualization in browser...")
        webbrowser.open(f'http://localhost:{port}/d3_parliamentary_viz.html')
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    start_server() 