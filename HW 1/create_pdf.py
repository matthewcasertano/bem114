import subprocess
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import textwrap
from PIL import Image as PILImage

def capture_output():
    """Run hw1.py and capture its output"""
    try:
        result = subprocess.run(['python3', 'hw1.py'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error running hw1.py: {str(e)}"

def read_code():
    """Read the contents of hw1.py"""
    try:
        with open('hw1.py', 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading hw1.py: {str(e)}"

def get_plot_files():
    """Get list of plot files from the plots directory"""
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        return []
    
    plot_files = []
    for file in os.listdir(plot_dir):
        if file.endswith('.png'):
            plot_files.append(os.path.join(plot_dir, file))
    return sorted(plot_files)  # Sort to ensure consistent order

def create_pdf():
    """Create a PDF containing the output, plots, and code"""
    # Create PDF document with smaller margins
    doc = SimpleDocTemplate(
        "hw1_submission.pdf",
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create a custom style for code and output with smaller font size
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,  # Smaller font size
        leading=10,  # Reduced line spacing
        spaceBefore=6,
        spaceAfter=6,
        textColor=colors.black,
        backColor=colors.lightgrey,
        borderWidth=1,
        borderColor=colors.black,
        borderPadding=3
    )
    
    # Create content
    content = []
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20
    )
    content.append(Paragraph("Homework 1 Submission", title_style))
    
    # Add output section
    content.append(Paragraph("Output:", styles['Heading2']))
    content.append(Spacer(1, 6))
    
    # Process output in chunks to handle long outputs
    output_text = capture_output()
    # Split output into lines and wrap long lines
    output_lines = []
    for line in output_text.split('\n'):
        wrapped_lines = textwrap.wrap(line, width=100)  # Adjust width as needed
        output_lines.extend(wrapped_lines)
    
    # Add output with proper formatting
    content.append(Preformatted('\n'.join(output_lines), code_style))
    content.append(PageBreak())  # Force a page break before plots section
    
    # Add plots section
    content.append(Paragraph("Plots:", styles['Heading2']))
    content.append(Spacer(1, 6))
    
    # Add each plot
    plot_files = get_plot_files()
    for plot_file in plot_files:
        # Get plot dimensions
        img = PILImage.open(plot_file)
        img_width, img_height = img.size
        
        # Calculate scaling to fit on page width
        page_width = letter[0] - 72  # Account for margins
        scale = page_width / img_width
        
        # Create scaled image
        img_width = img_width * scale
        img_height = img_height * scale
        
        # Add plot with caption
        content.append(Image(plot_file, width=img_width, height=img_height))
        content.append(Paragraph(f"Figure: {os.path.basename(plot_file)}", styles['Italic']))
        content.append(Spacer(1, 12))
    
    content.append(PageBreak())  # Force a page break before code section
    
    # Add code section
    content.append(Paragraph("Code:", styles['Heading2']))
    content.append(Spacer(1, 6))
    
    # Process code in chunks to handle long code
    code_text = read_code()
    # Split code into lines and wrap long lines
    code_lines = []
    for line in code_text.split('\n'):
        wrapped_lines = textwrap.wrap(line, width=100)  # Adjust width as needed
        code_lines.extend(wrapped_lines)
    
    # Add code with proper formatting
    content.append(Preformatted('\n'.join(code_lines), code_style))
    
    # Build PDF
    doc.build(content)
    print("PDF created successfully: hw1_submission.pdf")

if __name__ == "__main__":
    create_pdf() 