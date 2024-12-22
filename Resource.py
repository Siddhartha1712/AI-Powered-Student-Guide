import requests
from bs4 import BeautifulSoup

def fetch_res(topic):
    try:
        search_query = topic.replace(" ", "+") + "+study+materials"
        search_url = f"https://html.duckduckgo.com/html/?q={search_query}"

        # Send an HTTP GET request to the search URL
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(search_url, headers=headers)
        if response.status_code != 200:
            print("Failed to fetch data. Please check your internet connection or try again later.")
            return []

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all relevant links
        results = soup.find_all('a', class_='result__a')
        if not results:
            print("No study materials found. Please try another topic.")
            return []

        materials = []
        for result in results[:10]:  # Limit to the top 10 results
            title = result.get_text()
            link = result['href']
            materials.append({"name": title, "link": link})

        return materials

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the materials: {e}")
        return []
    except Exception as e:
        print(f"Error parsing the content: {e}")
        return []
if __name__ == "__main__":
    topic = input("Enter the name of the topic: ")
    material = fetch_res(topic)
    print(material)