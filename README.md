# AI-Powered Student Guide

## Overview
The AI-Powered Student Guide is a prototype web application designed to assist students with their studies. This guide offers functionalities such as providing information on topics, generating study roadmaps, summarizing text, and offering additional resources. The application utilizes Python-based modules and is powered by Flask as its backend framework.

---

## Features

### 1. **Information Providing**
- Module: `definition.py`
- Description: Fetches detailed topic-related content using web scraping techniques.

### 2. **Study Roadmap Generation**
- Module: `roadmap.py`
- Description: Creates a structured study roadmap to help students achieve their learning goals efficiently.

### 3. **Resource Providing**
- Module: `resource.py`
- Description: Gathers additional study resources like articles, videos, and reference materials via web scraping.

### 4. **Text Summarization**
- Module: `summarize.py`
- Description: Summarizes large text inputs into concise and meaningful summaries.

---

## Project Architecture

```
project-directory
|-- app.py                # Main Flask application
|-- definition.py         # Information-providing module
|-- roadmap.py            # Study roadmap generation module
|-- resource.py           # Resource-providing module
|-- summarize.py          # Text summarization module
|-- static/               # Static files (CSS, JS, images)
|-- templates/            # HTML templates
|-- README.md             # Project documentation (this file)
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/ai-powered-student-guide.git
   cd ai-powered-student-guide
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv env
   env\Scripts\activate     # For Windows
   ```

3. **Run the Flask application:**
   ```bash
   python app.py
   ```

4. **Access the web application:**
   Open your browser and navigate to `http://127.0.0.1:5000`

---

## Usage

### Endpoints
- **Home Page:** `/`
- **Information Providing:** `/definition`
- **Study Roadmap Generation:** `/roadmap`
- **Resource Providing:** `/resource`
- **Text Summarization:** `/summarize`

### Workflow
1. Navigate to the desired module's endpoint.
2. Input the required data (e.g., topic for `definition` or text for `summarize`).
3. View the output and utilize the provided information.

---

## Dependencies

- Python 3.8+
- Flask
- BeautifulSoup4 (for web scraping)
- Requests (for HTTP requests)
- Any other specific dependencies used in your project should be manually installed.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Commit your changes and push to your fork.
4. Submit a pull request explaining your changes.

---

## Contact

For questions, suggestions, or feedback, please contact:
- **Your Name**
- Email: its.trashi.17@gamil.com

