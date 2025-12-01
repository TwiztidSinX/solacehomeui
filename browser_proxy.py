"""
Browser Proxy Server for SolaceOS
Bypasses iframe restrictions and enables agentic browser control
"""

from flask import Flask, request, Response
import requests
from bs4 import BeautifulSoup
import re

app = Flask(__name__)

# JavaScript bridge injected into all pages
CONTROL_BRIDGE_JS = """
<script>
// SolaceOS Browser Control Bridge
window.solaceControl = {
    // Click an element by selector
    click: function(selector) {
        const element = document.querySelector(selector);
        if (element) {
            element.click();
            return { success: true, message: 'Clicked ' + selector };
        }
        return { success: false, message: 'Element not found: ' + selector };
    },

    // Fill a form field
    fill: function(selector, value) {
        const element = document.querySelector(selector);
        if (element) {
            element.value = value;
            element.dispatchEvent(new Event('input', { bubbles: true }));
            element.dispatchEvent(new Event('change', { bubbles: true }));
            return { success: true, message: 'Filled ' + selector };
        }
        return { success: false, message: 'Element not found: ' + selector };
    },

    // Get page content
    getContent: function() {
        return {
            title: document.title,
            url: window.location.href,
            text: document.body.innerText,
            html: document.body.innerHTML
        };
    },

    // Get all links
    getLinks: function() {
        return Array.from(document.querySelectorAll('a')).map(a => ({
            text: a.innerText,
            href: a.href
        }));
    },

    // Get all form fields
    getFormFields: function() {
        return Array.from(document.querySelectorAll('input, select, textarea')).map(field => ({
            name: field.name,
            type: field.type,
            value: field.value,
            id: field.id,
            placeholder: field.placeholder
        }));
    },

    // Scroll to element
    scrollTo: function(selector) {
        const element = document.querySelector(selector);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            return { success: true };
        }
        return { success: false, message: 'Element not found' };
    },

    // Execute custom JavaScript
    exec: function(code) {
        try {
            return { success: true, result: eval(code) };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }
};

// Notify parent that bridge is ready
console.log('[SolaceOS] Browser control bridge loaded');
</script>
"""

@app.route('/proxy')
def proxy():
    """
    Proxy a website, strip frame-blocking headers, inject control bridge
    Usage: http://localhost:8001/proxy?url=https://example.com
    """
    target_url = request.args.get('url')

    if not target_url:
        return "Error: No URL provided. Use ?url=https://example.com", 400

    # Add protocol if missing
    if not target_url.startswith('http'):
        target_url = 'https://' + target_url

    try:
        # Fetch the target page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        response = requests.get(target_url, headers=headers, timeout=10)

        # Get content type
        content_type = response.headers.get('Content-Type', '')

        # Only modify HTML pages
        if 'text/html' in content_type:
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Inject control bridge at the start of <head> or <body>
            if soup.head:
                # Create script tag
                script_tag = soup.new_tag('script')
                script_tag.string = CONTROL_BRIDGE_JS
                soup.head.insert(0, script_tag)
            elif soup.body:
                script_tag = soup.new_tag('script')
                script_tag.string = CONTROL_BRIDGE_JS
                soup.body.insert(0, script_tag)

            # Fix relative URLs to absolute
            base_url = '/'.join(target_url.split('/')[:3])

            # Fix links
            for tag in soup.find_all(['a', 'link']):
                if tag.get('href'):
                    href = tag['href']
                    if href.startswith('/'):
                        tag['href'] = base_url + href
                    elif not href.startswith('http'):
                        tag['href'] = target_url.rsplit('/', 1)[0] + '/' + href

            # Fix images
            for tag in soup.find_all('img'):
                if tag.get('src'):
                    src = tag['src']
                    if src.startswith('/'):
                        tag['src'] = base_url + src
                    elif not src.startswith('http'):
                        tag['src'] = target_url.rsplit('/', 1)[0] + '/' + src

            # Fix scripts
            for tag in soup.find_all('script'):
                if tag.get('src'):
                    src = tag['src']
                    if src.startswith('/'):
                        tag['src'] = base_url + src
                    elif not src.startswith('http'):
                        tag['src'] = target_url.rsplit('/', 1)[0] + '/' + src

            # Fix CSS
            for tag in soup.find_all('link', rel='stylesheet'):
                if tag.get('href'):
                    href = tag['href']
                    if href.startswith('/'):
                        tag['href'] = base_url + href
                    elif not href.startswith('http'):
                        tag['href'] = target_url.rsplit('/', 1)[0] + '/' + href

            content = str(soup)
        else:
            content = response.content

        # Create response with stripped headers
        proxy_response = Response(content)

        # Copy safe headers
        for header, value in response.headers.items():
            # Skip frame-blocking headers
            if header.lower() in ['x-frame-options', 'content-security-policy', 'content-security-policy-report-only']:
                continue
            # Skip headers that cause issues
            if header.lower() in ['content-encoding', 'content-length', 'transfer-encoding']:
                continue
            proxy_response.headers[header] = value

        # Add CORS headers
        proxy_response.headers['Access-Control-Allow-Origin'] = '*'
        proxy_response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        proxy_response.headers['Access-Control-Allow-Headers'] = '*'

        return proxy_response

    except requests.exceptions.Timeout:
        return f"Error: Request to {target_url} timed out", 504
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch {target_url}: {str(e)}", 500
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'ok', 'service': 'SolaceOS Browser Proxy'}

if __name__ == '__main__':
    print("=" * 50)
    print("SolaceOS Browser Proxy Server")
    print("=" * 50)
    print("Starting on http://localhost:8001")
    print("Usage: http://localhost:8001/proxy?url=https://example.com")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8001, debug=True)
