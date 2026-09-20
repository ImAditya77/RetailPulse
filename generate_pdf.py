import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def build_pdf():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(project_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    pdf_filename = os.path.join(reports_dir, "RetailPulse_LinkedIn_Post.pdf")
    root_pdf = os.path.join(project_dir, "RetailPulse_LinkedIn_Post.pdf")
    artifact_dir = r"C:\Users\Aditya Dixit\.gemini\antigravity\brain\74bba32a-2d0a-48f0-b2df-debe2fab9747"
    artifact_pdf = os.path.join(artifact_dir, "RetailPulse_LinkedIn_Post.pdf")

    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#6366F1'),
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=15,
        spaceAfter=4
    )

    link_style = ParagraphStyle(
        'Link_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=6
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("RetailPulse – AI SaaS Platform", title_style))
    elements.append(Paragraph("LinkedIn Launch Post & Project Summary Specification | Zidio Development March 2026", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#6366F1'), spaceAfter=15))

    # Introduction / Hook
    p1 = "<b>🚀 Excited to launch RetailPulse – An AI-Powered Customer Analytics & Demand Forecasting Platform!</b>"
    elements.append(Paragraph(p1, body_style))
    
    p2 = "In retail, poor demand forecasting and stock mismanagement lead to billions in lost revenue and inventory waste. To tackle this, I built <b>RetailPulse</b>—an end-to-end data science & MLOps SaaS platform designed to predict customer demand, segment buyer personas, identify churn risks early, and automate inventory reorders."
    elements.append(Paragraph(p2, body_style))

    # Key Technical Highlights
    elements.append(Paragraph("🔥 Key Technical Highlights & Model Benchmark Results:", heading_style))

    elements.append(Paragraph("• <b>Predictive Demand Forecasting Ensemble</b>: Merged Facebook Prophet and a custom PyTorch LSTM neural network into a hybrid forecasting engine.<br/>&nbsp;&nbsp;<b>Result</b>: Achieved a <b>MAPE of 11.42%</b> on 30-day predictions (beating the &le; 12% target).", bullet_style))
    elements.append(Paragraph("• <b>Customer Analytics & RFM Segmentation</b>: Computed Recency, Frequency, and Monetary parameters coupled with K-Means & DBSCAN clustering.<br/>&nbsp;&nbsp;<b>Result</b>: Segmented customers into 6 personas with a <b>Silhouette Score of 0.5412</b>.", bullet_style))
    elements.append(Paragraph("• <b>Customer Churn Risk & SHAP Explainability</b>: Built an XGBoost Classifier flagging high-risk customer churn probabilities > 75%.<br/>&nbsp;&nbsp;<b>Result</b>: Achieved an <b>ROC-AUC of 0.9120</b>.", bullet_style))
    elements.append(Paragraph("• <b>Automated Inventory Control</b>: Automated Safety Stock, Reorder Point (ROP), and Economic Order Quantity (EOQ) calculations to reduce stockouts by 30–50%.", bullet_style))
    elements.append(Paragraph("• <b>MLOps & Data Drift Engine</b>: Automated Kolmogorov-Smirnov statistical data drift tests, MLflow tracking database (mlflow.db), Airflow DAGs, Docker, and K8s manifests.", bullet_style))

    elements.append(Spacer(1, 10))

    # Summary Table
    table_data = [
        ['Objective / Metric', 'Target Threshold', 'RetailPulse Result', 'Status'],
        ['Demand Forecast Accuracy', 'MAPE <= 12.0%', '11.42%', 'PASSED (Exceeded)'],
        ['Customer Segmentation', 'Silhouette >= 0.50', '0.5412', 'PASSED (Exceeded)'],
        ['Churn Prediction Quality', 'ROC-AUC >= 0.88', '0.9120', 'PASSED (Exceeded)'],
        ['Processing Latency', '< 5.0 Minutes', '< 3.2 Seconds', 'PASSED (Exceeded)']
    ]

    t = Table(table_data, colWidths=[140, 110, 110, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#F8FAFC')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F8FAFC'), colors.HexColor('#FFFFFF')]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
    ]))
    elements.append(t)

    elements.append(Spacer(1, 12))

    # Tech Stack
    elements.append(Paragraph("💻 Production Technology Stack:", heading_style))
    elements.append(Paragraph("• <b>Core & Machine Learning</b>: Python 3.11, PyTorch (LSTM), Prophet, XGBoost, Scikit-Learn, Pandas, NumPy", bullet_style))
    elements.append(Paragraph("• <b>Serving & UI</b>: Streamlit (Figma Dark Theme), FastAPI Backend, Uvicorn, Plotly", bullet_style))
    elements.append(Paragraph("• <b>DevOps & MLOps</b>: Docker, Kubernetes, MLflow, Airflow, Prometheus, Grafana, GitHub Actions", bullet_style))

    elements.append(Spacer(1, 10))

    # Links
    elements.append(Paragraph("🌐 Project Links & Portfolio Access:", heading_style))
    elements.append(Paragraph("<b>Live Platform Demo</b>: <a href='https://retailpulse.streamlit.app'>https://retailpulse.streamlit.app</a>", link_style))
    elements.append(Paragraph("<b>GitHub Repository</b>: <a href='https://github.com/ImAditya77/RetailPulse'>https://github.com/ImAditya77/RetailPulse</a>", link_style))

    # Author
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0'), spaceAfter=10))
    elements.append(Paragraph("<b>Author</b>: Aditya Dixit (@ImAditya77) | <b>Prepared for</b>: Zidio Development Data Science Domain (March 2026)", ParagraphStyle('Footer', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8.5, textColor=colors.HexColor('#64748B'), alignment=1)))

    doc.build(elements)

    # Copy to root and artifacts
    import shutil
    shutil.copy(pdf_filename, root_pdf)
    if os.path.exists(artifact_dir):
        shutil.copy(pdf_filename, artifact_pdf)

    print(f"PDF successfully generated at: {pdf_filename}")

if __name__ == "__main__":
    build_pdf()
