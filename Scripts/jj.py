def generate_notice_html():
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <style>
        @page {
            size: A4;
            margin: 0;
        }
        body {
            width: 210mm;
            height: 297mm;
            margin: 0 auto;
            padding: 20mm;
            background-color: white;
            font-family: Arial, sans-serif;
            box-sizing: border-box;
        }
        .notice {
            border: 2px solid black;
            padding: 20px;
            height: calc(100% - 4px);
            position: relative;
        }
        h1, h2 {
            text-align: center;
            margin: 10px 0;
        }
        h1 {
            font-size: 24px;
            border-bottom: 2px solid black;
            padding-bottom: 10px;
        }
        h2 {
            font-size: 18px;
        }
        .details {
            border: 1px solid black;
            padding: 15px;
            margin: 15px 0;
        }
        .competition {
            margin: 10px 0;
            padding-left: 20px;
        }
        .important {
            font-weight: bold;
            border: 1px solid black;
            padding: 10px;
            margin: 10px 0;
            background-color: #f8f8f8;
        }
    </style>
</head>
<body>
    <div class="notice">
        <h1>NOTICE</h1>
        <h2>Annual English Literary Competition</h2>
        <h2>Academic Session 2024-25</h2>
        
        <div class="details">
            <div class="important">
                Date: 18th February 2025<br>
                Time: 12:00 NOON onwards<br>
                Venue: Room Number 210
            </div>

            <h2>Competitions:</h2>
            <div class="competition">
                1. Essay Writing - Dr. Sankalp Jogi<br>
                Topic: "The Significance of Guru-Shishya Bonding in the Gurukul System of Education"<br>
                Word limit: 500 words
            </div>
            <div class="competition">
                2. Poster Competition - Ms. Shivali Rai<br>
                Topic: Indian Knowledge Tradition
            </div>
            <div class="competition">
                3. Quiz Competition - Mrs. Rachna Thakur<br>
                Mode: On the spot
            </div>
            <div class="competition">
                4. Picture Composition - Dr. Usha Masram<br>
                Mode: On the spot (200-250 words)
            </div>
        </div>

        <div class="important">
            Eligibility: BSc, BCA, MSc, and MA English Literature students<br>
            Registration Deadline: 15th February 2025<br>
            Contact: Respective event incharge or Department of English
        </div>
    </div>
</body>
</html>'''

    # Write the HTML content to a file
    with open('notice.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    generate_notice_html()
    print("Notice HTML file has been generated as 'notice.html'")