import html.parser

class P(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.texts = []
        self.current_tags = []
        
    def handle_starttag(self, tag, attrs):
        self.current_tags.append((tag, dict(attrs).get('class', '')))
        
    def handle_endtag(self, tag):
        if self.current_tags:
            self.current_tags.pop()
            
    def handle_data(self, data):
        data = data.strip()
        if len(data) > 50 and self.current_tags:
            tag, cls = self.current_tags[-1]
            self.texts.append(f"<{tag} class='{cls}'>: {data[:100]}...")

p = P()
p.feed(open('review_page.html', encoding='utf-8').read())
with open('review_texts.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(p.texts))

