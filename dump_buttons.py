import html.parser

class MyHTMLParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def handle_starttag(self, tag, attrs):
        if tag == 'input':
            attr_dict = dict(attrs)
            self.inputs.append(f"id: {attr_dict.get('id', '')}, name: {attr_dict.get('name', '')}, class: {attr_dict.get('class', '')}, type: {attr_dict.get('type', '')}")

parser = MyHTMLParser()
with open('page.html', encoding='utf-8') as f:
    parser.feed(f.read())

with open('debug_output.txt', 'w', encoding='utf-8') as f:
    f.write("Inputs:\n" + "\n".join(parser.inputs))
