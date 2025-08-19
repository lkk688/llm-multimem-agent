#!/bin/bash

# Complete Pandoc + reveal.js Presentation Generator
# Converts physical_ai_autonomous_driving.md to reveal.js presentation

echo "🚀 Creating reveal.js presentation with Pandoc..."
echo "================================================"

# Check dependencies
echo "Checking dependencies..."

if ! command -v pandoc &> /dev/null; then
    echo "❌ Error: pandoc is not installed"
    echo "Please install pandoc:"
    echo "  macOS: brew install pandoc"
    echo "  Linux: sudo apt-get install pandoc"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed"
    exit 1
fi

echo "✅ Dependencies check passed"

# File paths
INPUT_FILE="docs/physical_ai_autonomous_driving.md"
PREPROCESSED_FILE="physical_ai_slides.md"
OUTPUT_FILE="physical_ai_presentation.html"
CONFIG_FILE="pandoc_revealjs_config.yaml"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file $INPUT_FILE not found"
    exit 1
fi

# Step 1: Preprocess the markdown file
echo "📝 Step 1: Preprocessing markdown for slides..."
python3 preprocess_for_slides.py

if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed"
    exit 1
fi

echo "✅ Preprocessing completed"

# Step 2: Convert with pandoc using configuration file
echo "🔄 Step 2: Converting with Pandoc + reveal.js..."

pandoc "$PREPROCESSED_FILE" \
    --defaults="$CONFIG_FILE" \
    -t revealjs \
    -s \
    -o "$OUTPUT_FILE" \
    --slide-level=2 \
    --highlight-style=github \
    --mathjax \
    --variable revealjs-url=https://unpkg.com/reveal.js@4.3.1/ \
    --variable theme=black \
    --variable transition=slide \
    --variable backgroundTransition=fade \
    --variable hash=true \
    --variable controls=true \
    --variable progress=true \
    --variable center=true \
    --variable touch=true \
    --variable hideAddressBar=true \
    --variable width=1280 \
    --variable height=720 \
    --metadata title="Physical AI and LLMs in Autonomous Driving" \
    --metadata author="AI Research Team" \
    --metadata date="$(date +'%Y-%m-%d')"

if [ $? -eq 0 ]; then
    echo "✅ Pandoc conversion successful!"
else
    echo "❌ Pandoc conversion failed. Trying alternative approach..."
    
    # Fallback: simpler pandoc command
    pandoc "$PREPROCESSED_FILE" \
        -t revealjs \
        -s \
        -o "$OUTPUT_FILE" \
        --slide-level=2 \
        --theme=black \
        --transition=slide \
        --highlight-style=github \
        --mathjax \
        --variable revealjs-url=https://unpkg.com/reveal.js@4.3.1/
    
    if [ $? -ne 0 ]; then
        echo "❌ Conversion failed completely. Check error messages above."
        exit 1
    fi
fi

# Step 3: Create a simple HTTP server script for viewing
echo "🌐 Step 3: Creating server script..."

cat > serve_pandoc_presentation.py << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def serve_presentation():
    # Change to the directory containing the presentation
    presentation_file = "physical_ai_presentation.html"
    
    if not Path(presentation_file).exists():
        print(f"Error: {presentation_file} not found")
        sys.exit(1)
    
    # Try different ports
    ports = [8000, 8001, 8002, 8003, 8080]
    
    for port in ports:
        try:
            with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
                print(f"🚀 Serving presentation at http://localhost:{port}/{presentation_file}")
                print(f"📱 Mobile access: http://[your-ip]:{port}/{presentation_file}")
                print("\n🎮 Presentation Controls:")
                print("  - Arrow keys: Navigate slides")
                print("  - F: Fullscreen mode")
                print("  - S: Speaker notes")
                print("  - O: Overview mode")
                print("  - ?: Help")
                print("\n⏹️  Press Ctrl+C to stop the server")
                
                # Open in browser
                webbrowser.open(f"http://localhost:{port}/{presentation_file}")
                
                # Start serving
                httpd.serve_forever()
                
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"Port {port} is busy, trying next port...")
                continue
            else:
                print(f"Error starting server on port {port}: {e}")
                continue
    
    print("❌ Could not start server on any available port")
    sys.exit(1)

if __name__ == "__main__":
    serve_presentation()
EOF

chmod +x serve_pandoc_presentation.py

echo "✅ Server script created"

# Step 4: Display results
echo ""
echo "🎉 Presentation creation completed!"
echo "================================================"
echo "📄 Files created:"
echo "  - $PREPROCESSED_FILE (preprocessed markdown)"
echo "  - $OUTPUT_FILE (reveal.js presentation)"
echo "  - serve_pandoc_presentation.py (local server)"
echo ""
echo "🚀 To view the presentation:"
echo "  1. Run: python3 serve_pandoc_presentation.py"
echo "  2. Or open $OUTPUT_FILE directly in a web browser"
echo ""
echo "🎮 Presentation features:"
echo "  - Responsive design optimized for different screen sizes"
echo "  - Dark theme with custom styling"
echo "  - Math equations support (MathJax)"
echo "  - Code syntax highlighting"
echo "  - Touch/swipe navigation for mobile devices"
echo "  - Speaker notes and overview mode"
echo ""
echo "📱 Navigation:"
echo "  - Arrow keys or swipe to navigate"
echo "  - Press 'f' for fullscreen"
echo "  - Press 's' for speaker notes"
echo "  - Press 'o' for overview mode"
echo "  - Press '?' for help"

# Clean up intermediate file
if [ -f "$PREPROCESSED_FILE" ]; then
    echo "🧹 Cleaning up intermediate files..."
    # Keep the preprocessed file for debugging
    # rm "$PREPROCESSED_FILE"
fi

echo "✅ Done!"