

# Dynamic Question Generation API

This project is a secure and robust Flask-based API that dynamically generates educational questions using Large Language Models (LLMs). It can interface with both Google's Gemini and OpenAI-compatible models like DeepSeek to produce a variety of question types based on provided content, book references, and specific generation rules. The application is containerized with Docker for easy deployment and scaling.

## Key Features

  * [cite\_start]**Dual LLM Support**: Seamlessly switch between Google Gemini and DeepSeek models for question generation by specifying the model in the API request[cite: 5].
  * [cite\_start]**Secure Authentication**: Endpoints are protected using JSON Web Tokens (JWT), requiring users to log in to access the generation capabilities[cite: 8].
  * [cite\_start]**Robust Data Validation**: Utilizes Marshmallow schemas to validate all incoming request data, ensuring data integrity and providing clear error messages[cite: 5, 8].
  * [cite\_start]**Rate Limiting**: Implements rate limiting on sensitive endpoints like `/login` to prevent abuse and enhance security[cite: 8].
  * [cite\_start]**Structured, Production-Ready Logging**: Employs `structlog` to produce structured, JSON-formatted logs for application events, access, errors, and security, facilitating easier monitoring and debugging[cite: 7].
  * [cite\_start]**Centralized Configuration**: Manages all configuration, including API keys and secrets, through environment variables for better security and flexibility[cite: 2].
  * [cite\_start]**Containerized Deployment**: Includes a multi-stage `Dockerfile` that builds a lightweight, secure image and runs the application in a production environment using a Gunicorn WSGI server[cite: 3].
  * **Custom Error Handling**: Global error handlers are set up to catch validation errors, service-level failures, and other exceptions, returning consistent JSON error responses.

## Technologies Used

The project is built with the following major technologies:

  * [cite\_start]**Backend Framework**: Flask [cite: 1]
  * [cite\_start]**Web Server**: Gunicorn [cite: 1]
  * [cite\_start]**Security**: Flask-JWT-Extended, Flask-Limiter [cite: 1]
  * [cite\_start]**AI Model SDKs**: google-generativeai, openai [cite: 1]
  * [cite\_start]**Data Validation**: marshmallow [cite: 1]
  * [cite\_start]**Logging**: structlog [cite: 1]
  * **Containerization**: Docker

## Project Structure

```
├── app/
│   ├── __init__.py         # Application factory, initializes Flask app and extensions
│   ├── config.py           # Configuration loader from environment variables
│   ├── extensions.py       # Initializes and configures Flask extensions (JWT, Limiter, etc.)
│   ├── logger.py           # Configures structured logging
│   ├── routes.py           # Defines API endpoints and view functions
│   ├── schemas.py          # Marshmallow schemas for data validation/serialization
│   └── services.py         # Contains business logic for interacting with LLMs
├── logs/                   # Directory for log files (created automatically)
│   ├── access/
│   ├── app/
│   ├── error/
│   └── security/
├── .env.example            # Example environment variables file
├── requirements.txt        # Python project dependencies
├── Dockerfile              # Instructions for building the production Docker image
└── run.py                  # Entry point to run the Flask application
```

## Setup and Installation

### 1\. Clone the Repository

```sh
git clone <repository-url>
cd <repository-directory>
```

### 2\. Create and Activate a Virtual Environment

```sh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3\. Install Dependencies

[cite\_start]Install all the required Python packages from the `requirements.txt` file[cite: 1].

```sh
pip install -r requirements.txt
```

## Configuration

The application requires several environment variables for its configuration.

1.  Create a `.env` file by copying the example file:

    ```sh
    cp .env.example .env
    ```

2.  [cite\_start]Update the `.env` file with your specific credentials and keys[cite: 2].

| Variable | Description |
| :--- | :--- |
| `FLASK_SECRET_KEY` | A random, secret key for signing Flask sessions. |
| `FLASK_DEBUG` | [cite\_start]Set to `True` for development, `False` for production[cite: 2]. |
| `JWT_SECRET_KEY` | [cite\_start]A random, secret key for signing JSON Web Tokens (JWTs)[cite: 2]. |
| `ADMIN_USERNAME` | [cite\_start]The username for logging into the API[cite: 2]. |
| `ADMIN_PASSWORD` | [cite\_start]The password for logging into the API[cite: 2]. |
| `GOOGLE_API_KEY` | [cite\_start]Your API key for the Google Gemini service[cite: 2]. |
| `DEEPSEEK_API_KEY` | [cite\_start]Your API key for the DeepSeek service[cite: 2]. |
| `CORS_ORIGINS` | [cite\_start]Comma-separated list of allowed origins for CORS requests[cite: 2]. |

## Running the Application

### Local Development

To run the application with the Flask development server:

```sh
flask run
```

The API will be available at `http://127.0.0.1:5000`.

### Production (using Gunicorn)

To run the application using the Gunicorn server as specified in the `Dockerfile`:

```sh
gunicorn --workers 4 --bind 0.0.0.0:5000 "run:app"
```

## API Endpoints

### 1\. User Login

Authenticates the user and returns a JWT access token.

  * **Endpoint**: `/login`
  * **Method**: `POST`
  * **Request Body**:
    ```json
    {
        "username": "your_admin_username",
        "password": "your_admin_password"
    }
    ```
  * **Success Response (200 OK)**:
    ```json
    {
        "access_token": "your.jwt.token"
    }
    ```

### 2\. Generate Questions

Generates questions based on the provided context and rules. This is a protected endpoint and requires authentication.

  * **Endpoint**: `/generate_questions`
  * **Method**: `POST`
  * **Headers**:
      * `Authorization: Bearer <your.jwt.token>`
  * **Request Body**:
    ```json
    {
        "module": "Linear Algebra",
        "content": "Vectors, vector spaces, dot product, cross product, matrices.",
        "Rules": [
            {
                "questionId": 1,
                "questionType": "multiple choice(MCQ)",
                "difficultyLevel": "Easy",
                "cognitiveLevel": "Knowledge",
                "mark": 2,
                "numberOfQuestions": 2
            }
        ],
        "BookDetails": [
            {
                "BookName": "Advanced Engineering Mathematics",
                "BookType": "Textbook"
            }
        ],
        "model": "gemini-pro"
    }
    ```
  * **Success Response (200 OK)**:
    ```json
    {
        "generated_questions": [
            {
                "question": "What is a vector?",
                "questionLatex": "What is a vector?",
                "answer": "A quantity having direction as well as magnitude.",
                "answerLatex": "A quantity having direction as well as magnitude.",
                "cognitiveLevel": "Knowledge",
                "difficultyLevel": "Easy",
                "mark": 2,
                "questionType": "multiple choice(MCQ)"
            }
        ]
    }
    ```

## Docker Deployment

The application can be easily built and run as a Docker container.

1.  **Build the Docker Image**:
    From the root directory of the project, run the following command. The `-t` flag tags the image for easier reference.

    ```sh
    docker build -t question-generation-api .
    ```

2.  **Run the Docker Container**:
    Run the container, mapping the container's port 5000 to the host's port 5000. You must also pass the environment variables from your `.env` file.

    ```sh
    docker run -p 5000:5000 --env-file .env question-generation-api
    ```

The API will now be accessible at `http://localhost:5000`.
