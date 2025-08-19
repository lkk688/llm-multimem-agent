#!/usr/bin/env python3
"""
Simple HTTP server to serve the reveal.js presentation locally.
Usage: python serve_presentation.py
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def serve_presentation(port=8000):
    """Serve the presentation on localhost."""
    # Change to the directory containing the presentation
    presentation_dir = Path(__file__).parent
    os.chdir(presentation_dir)
    
    # Create HTTP server
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Serving presentation at http://localhost:{port}/presentation.html")
            print(f"Press Ctrl+C to stop the server")
            
            # Automatically open browser
            webbrowser.open(f"http://localhost:{port}/presentation.html")
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Port {port} is already in use. Trying port {port + 1}...")
            serve_presentation(port + 1)
        else:
            print(f"Error starting server: {e}")
            sys.exit(1)

if __name__ == "__main__":
    serve_presentation()