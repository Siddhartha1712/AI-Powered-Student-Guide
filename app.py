from flask import Flask, render_template, request
from definiton import get_topic_content
from Resource import fetch_res
from roadmap import graph
from summarize import generate_summary

app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/def", methods=["GET", "POST"])
def definition():
    if request.method == "POST":
        topic = request.form.get("topic")
        if topic:
            # Example response from get_topic_content
            raw_content = get_topic_content(topic)

            # Convert raw content into structured data for paragraphs and lists
            content = []

            if isinstance(raw_content, str):
                paragraphs = raw_content.split('\n\n')  # Split by double newlines for paragraphs
                for para in paragraphs:
                    if '\n' in para:
                        # If there's a newline within, split into bullet points
                        points = para.split('\n')
                        content.append({"type": "list", "points": points})
                    else:
                        # Add as a paragraph
                        content.append({"type": "paragraph", "text": para})

            elif isinstance(raw_content, list):
                # If raw_content is already a list, add it as bullet points
                content.append({"type": "list", "points": raw_content})

            return render_template("def.html", topic=topic, content=content)

    return render_template("def.html", topic=None, content=None)




@app.route("/rod", methods=["GET", "POST"])
def roadmap():
    content = None  # Initialize the content to display in the template
    topic = None  # Initialize the topic variable

    if request.method == "POST":
        topic = request.form.get("topic")  # Get the topic from the form
        if topic:
            roadmap_tree = graph()  # Build the tree structure
            node = roadmap_tree.find_node(topic.lower())  # Find the node matching the topic
            if node:
                # Collect tree structure into a list instead of printing it
                roadmap_output = []

                def collect_tree_data(current_node, prefix="", is_last=True):
                    """
                    Recursive function to collect the tree structure.
                    """
                    line = f"{prefix}{'`-- ' if is_last else '|-- '}{current_node.data}"
                    roadmap_output.append(line)
                    new_prefix = prefix + ("    " if is_last else "|   ")
                    for i, child in enumerate(current_node.child):
                        collect_tree_data(child, new_prefix, i == len(current_node.child) - 1)

                # Start collecting the tree structure
                collect_tree_data(node)
                content = roadmap_output
            else:
                content = [f"No roadmap found for '{topic}'."]

    return render_template("rod.html", topic=topic, content=content)

@app.route("/res", methods=["GET", "POST"])
def resource():
    resources = None  # Initialize resources
    topic = None  # Initialize topic

    if request.method == "POST":
        topic = request.form.get("topic")  # Get the topic from the form
        if topic:
            resources = fetch_res(topic)[:10]  # Fetch only the top 10 results

    return render_template("res.html", topic=topic, resources=resources)



@app.route("/sum", methods=["GET", "POST"])
def summarize():
    if request.method == "POST":
        text = request.form.get("text")  # Get the essay text from the form
        num_sentences = int(request.form.get("num_sentences", 3))  # Default to 3 sentences
        if text:
            # Generate the summary
            summary = generate_summary(text, num_sentences)
            return render_template("sum.html", text=text, summary=summary)  # Return text and summary to the template
    # Render the form if no POST data is available
    return render_template("sum.html", text=None, summary=None)

if __name__ == "__main__":
    app.run(debug=True)
