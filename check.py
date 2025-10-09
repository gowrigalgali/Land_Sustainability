from fpdf import FPDF

# Create instance of FPDF class
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Add a page
pdf.add_page()

# Set title
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "Green Cover Index (GCI) Prediction Models", ln=True, align='C')
pdf.ln(10)

# Set font for body text
pdf.set_font("Arial", '', 12)

# Add model descriptions
models = {
    "Linear Regression": [
        ("Purpose", "Linear Regression predicts a continuous target variable based on a linear relationship with one or more predictors."),
        ("How it Works", "Fits a straight line through historical GCI vs Year data and uses the line to predict future GCI values."),
        ("What it Predicts", "Forecasts GCI for the next decade (2025-2035) assuming an approximately linear trend.")
    ],
    "Support Vector Machine (SVM) Regression": [
        ("Purpose", "SVM regression captures non-linear trends in the data using kernel functions."),
        ("How it Works", "Maps Year data into a higher-dimensional space, fits a function within a margin of error, minimizing prediction errors."),
        ("What it Predicts", "Flexible GCI predictions capturing subtle non-linear patterns.")
    ],
    "Random Forest Regression": [
        ("Purpose", "Random Forest uses multiple decision trees to improve prediction accuracy and reduce overfitting."),
        ("How it Works", "Trains multiple decision trees on subsets of data and averages their predictions."),
        ("What it Predicts", "Robust GCI forecasts for irregular or noisy historical trends.")
    ],
    "Deep Learning Regression (Optional)": [
        ("Purpose", "Neural networks model highly complex, non-linear patterns."),
        ("How it Works", "Feedforward neural network learns the relationship between Year and historical GCI through iterative optimization."),
        ("What it Predicts", "GCI values accounting for complex trends; improves with more data and features.")
    ]
}

for model_name, details in models.items():
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, model_name, ln=True)
    pdf.set_font("Arial", '', 12)
    for heading, text in details:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"{heading}:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 7, text)
    pdf.ln(5)

# Add Performance Metrics table
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Performance Metrics (Summary Table)", ln=True)
pdf.ln(2)

# Table headers
pdf.set_font("Arial", 'B', 12)
col_widths = [50, 30, 30, 30]
headers = ["Model", "R²", "MAE", "RMSE"]
for i, header in enumerate(headers):
    pdf.cell(col_widths[i], 8, header, border=1, align='C')
pdf.ln()

# Example table data (replace 0.xxx with real metrics)
pdf.set_font("Arial", '', 12)
table_data = [
    ["Linear Regression", "0.xxx", "0.xxx", "0.xxx"],
    ["SVM Regression", "0.xxx", "0.xxx", "0.xxx"],
    ["Random Forest", "0.xxx", "0.xxx", "0.xxx"],
    ["Deep Learning (Optional)", "0.xxx", "0.xxx", "0.xxx"]
]

for row in table_data:
    for i, item in enumerate(row):
        pdf.cell(col_widths[i], 8, item, border=1, align='C')
    pdf.ln()

# Save PDF
pdf.output("GCI_Model_Report.pdf")
print("PDF report generated: GCI_Model_Report.pdf")
