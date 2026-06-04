#!/usr/bin/env python3
"""Dev server for the static UI — same idea as `manage.py runserver`, no Django."""
import argparse
import http.server
import os
import socketserver

UI_DIR = os.path.join(os.path.dirname(__file__), "ui")
DEFAULT_PORT = 8000


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the LLM Validation UI (static files).")
    parser.add_argument(
        "port",
        nargs="?",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port (default: {DEFAULT_PORT})",
    )
    args = parser.parse_args()
    os.chdir(UI_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"Serving UI at http://127.0.0.1:{args.port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
