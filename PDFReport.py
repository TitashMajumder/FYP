# File: PDFReport.py
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf_report(summary, output_file="tree_health_report.pdf"):
     doc = SimpleDocTemplate(output_file)
     styles = getSampleStyleSheet()
     elements = []

     elements.append(Paragraph("Tree Health Report", styles['Title']))
     elements.append(Spacer(1, 12))

     for cluster, row in summary.iterrows():
          text = f"Cluster {cluster}: {row.to_dict()}"
          elements.append(Paragraph(text, styles['Normal']))
          elements.append(Spacer(1, 6))

     doc.build(elements)
     print(f"PDF report saved → {output_file}")
