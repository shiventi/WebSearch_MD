from http.server import BaseHTTPRequestHandler, HTTPServer
from pymongo import MongoClient
import json

class MyRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/get_data':
            # Connect to MongoDB
            client = MongoClient("mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000")
            database = client.questiondb
            collection = database.question

            # Retrieve data from MongoDB
            data = list(collection.find({}, {'_id': 0}))

            # Send JSON response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            # Handle other paths
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Not Found')

def run():
    port = 8000
    server_address = ('', port)
    httpd = HTTPServer(server_address, MyRequestHandler)
    print(f'Starting server on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run()