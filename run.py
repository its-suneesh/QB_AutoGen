# run.py

from app import create_app

app = create_app()

if __name__ == "__main__":
    # Use Gunicorn or another WSGI server in production
    app.run(host='0.0.0.0', port=5000, debug=True)
