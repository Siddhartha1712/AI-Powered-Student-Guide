import requests
from bs4 import BeautifulSoup

def get_topic_content(topic):
    try:
        topic = topic.replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{topic}"
        response = requests.get(url)
        if response.status_code != 200:
            return f"Unable to retrieve the page for '{topic}'. Check the topic or try again later."

        soup = BeautifulSoup(response.content, 'html.parser')

        paragraphs = soup.find_all('p')

        content = []
        for paragraph in paragraphs:
            text = paragraph.text.strip()
            if text and len(text) > 50:
                content.append(text)
            if len(content) >= 6: 
                break

        return "\n\n".join(content) if content else f"No detailed information found for '{topic}'."

    except Exception as e:
        return f"An error occurred: {e}"

