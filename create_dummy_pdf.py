from reportlab.pdfgen import canvas

def create_dummy_pdf(filename="dummy_resume.pdf"):
    c = canvas.Canvas(filename)
    c.drawString(100, 750, "John Doe")
    c.drawString(100, 730, "Software Engineer with 5 years of experience.")
    c.drawString(100, 710, "Skills: Python, Streamlit, SQLite, AI.")
    c.drawString(100, 690, "Education: Bachelor in Computer Science.")
    c.save()
    print(f"Created {filename}")

if __name__ == "__main__":
    create_dummy_pdf()
