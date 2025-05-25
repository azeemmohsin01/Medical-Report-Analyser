import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytesseract
import cv2
import re
from PyPDF2 import PdfReader
from io import BytesIO

# Configure Tesseract path (if needed) - UNCOMMENT AND SET IF TESSERACT IS NOT IN PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Streamlit UI Setup
st.set_page_config(page_title="Medical Report Analyzer", layout="wide")
st.title("📊 Medical Report Analyzer")
st.markdown("""
<style>
    .report-title { font-size: 24px; color: #2e86c1; }
    .section-header { font-size: 20px; color: #28b463; margin-top: 20px; }
    .highlight { background-color: #f8f9fa; padding: 12px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# File Uploader - ADD CSV and XLSX types
uploaded_file = st.file_uploader("Upload Medical Report",
                                 type=["pdf", "jpg", "jpeg", "png", "tiff", "bmp", "csv", "xlsx"],
                                 help="Supported formats: PDF, JPG, PNG, TIFF, BMP, CSV, XLSX")

def preprocess_image(image):
    """Enhance image quality for better OCR"""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Denoising - more robust denoising can be helpful
    denoised = cv2.fastNlMeansDenoising(thresh, h=10, templateWindowSize=7, searchWindowSize=21)
    
    return denoised

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyPDF2"""
    text = ""
    try:
        pdf_reader = PdfReader(BytesIO(pdf_file.read()))
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    return text

# **NEW FUNCTION FOR TABULAR DATA**
def extract_text_from_tabular(df):
    """
    Converts a Pandas DataFrame into a single string, mimicking text extraction.
    This is a workaround to use existing regex patterns.
    For structured data, a different extraction approach would be better.
    """
    text_content = ""
    for col in df.columns:
        text_content += f"{col}:\n" # Add column name as a header
        for item in df[col].dropna().astype(str): # Convert all to string, drop NaNs
            text_content += f"{item}\n"
    return text_content.lower() # Convert to lower case for regex matching

def extract_medical_values(text):
    """
    More robust value extraction with multiple pattern matching for all common tests.
    Prioritizes patterns that explicitly mention units or are more specific.
    """
    # Lowercase the entire text for case-insensitive matching
    text_lower = text.lower()

    # Enhanced patterns to match different report formats
    # The order of patterns within each list matters: more specific/reliable patterns first.
    # The first capture group (the first parenthesis) should always capture the numerical value.
    patterns = {
        "Hemoglobin": [
            r"hemoglobin\s*\(?hb\)?[\s:;=\-]*(\d+\.?\d*)\s*g/dl", # Specific to (Hb) and unit
            r"hemoglobin[\s:;=\-]*(\d+\.?\d*)\s*g/dl",          # Hemoglobin with unit
            r"hb[\s:;=\-]*(\d+\.?\d*)\s*g/dl",                  # Hb with unit
            r"hgb[\s:;=\-]*(\d+\.?\d*)\s*g/dl",                 # HGB with unit
            r"hemoglobin\s*\(?hb\)?.*?(\d+\.?\d*)",             # Hemoglobin (Hb) general, non-greedy match
            r"hemoglobin.*?(\d+\.?\d*)",                       # General Hemoglobin match
            r"hb.*?(\d+\.?\d*)",                               # General Hb match
            r"hgb.*?(\d+\.?\d*)"                               # General HGB match
        ],
        "Blood Sugar": [
            r"(fasting\s*blood\s*sugar|f\.?\s*b\.?\s*s\.?|blood\s*sugar|glucose|bs|r\.?\s*b\.?\s*s\.?|random\s*blood\s*sugar|post\s*prandial\s*blood\s*sugar|ppbs)[\s:;=\-]*(\d+\.?\d*)\s*mg/dl", # Common FBS/RBS/PPBS with mg/dl unit
            r"(fasting\s*blood\s*sugar|f\.?\s*b\.?\s*s\.?|blood\s*sugar|glucose|bs|r\.?\s*b\.?\s*s\.?|random\s*blood\s*sugar|post\s*prandial\s*blood\s*sugar|ppbs)[\s:;=\-]*(\d+\.?\d*)\s*mmol/l", # Common FBS/RBS/PPBS with mmol/l unit
            r"glucose.*?(\d+\.?\d*)",                           # General Glucose
            r"blood\s*sugar.*?(\d+\.?\d*)",                     # General Blood Sugar
            r"bs.*?(\d+\.?\d*)",                                # General BS
            r"f\.?b\.?s\.?.*?(\d+\.?\d*)",                      # General FBS
            r"r\.?b\.?s\.?.*?(\d+\.?\d*)",                      # General RBS
            r"ppbs.*?(\d+\.?\d*)"                               # General PPBS
        ],
        "Cholesterol": [
            r"(total\s*cholesterol|cholesterol|chol)[\s:;=\-]*(\d+\.?\d*)\s*mg/dl", # Total Cholesterol with unit
            r"(total\s*cholesterol|cholesterol|chol)[\s:;=\-]*(\d+\.?\d*)",     # Total Cholesterol general
            r"lipid\s*profile.*?cholesterol.*?(\d+\.?\d*)", # Catches "Lipid Profile ... Cholesterol: XXX"
            r"cholesterol.*?(\d+\.?\d*)",                         # General Cholesterol
            r"chol.*?(\d+\.?\d*)"                                 # General Chol
        ],
        "WBC": [
            r"(total\s+wbc\s+count|white blood cells|wbc|leukocytes)[\s:;=\-]*(\d+\.?\d*)\s*k/μl", # Added total wbc count, with unit
            r"(total\s+wbc\s+count|wbc)[\s:;=\-]*(\d+\.?\d*)\s*x\s*10\^3/ul", # Specific unit format for WBC
            r"(total\s+wbc\s+count|wbc)[\s:;=\-]*(\d+\.?\d*)\s*(k/ul|x10\^9/l)", # Handle k/ul or x10^9/l
            r"(total\s+wbc\s+count|white blood cells|wbc|leukocytes).*?(\d+\.?\d*)" # General WBC
        ],
        "Platelets": [
            r"(platelets|plt|platelet\s*count|plt\s*count)[\s:;=\-]*(\d+\.?\d*)\s*k/μl", # Platelets with k/μl unit
            r"(platelets|plt|platelet\s*count|plt\s*count)[\s:;=\-]*(\d+\.?\d*)\s*x\s*10\^3/ul", # Platelets with x 10^3/uL unit
            r"(platelets|plt|platelet\s*count|plt\s*count)[\s:;=\-]*(\d+\.?\d*)\s*(k/ul|x10\^9/l)", # Platelets with various cell count units
            r"(platelets|plt|platelet\s*count|plt\s*count).*?(\d+\.?\d*)" # General Platelets
        ]
    }
    
    results = {}
    for test, test_patterns in patterns.items():
        value = None
        for pattern_str in test_patterns:
            match = re.search(pattern_str, text_lower, re.IGNORECASE)
            if match:
                for i in range(1, len(match.groups()) + 1):
                    captured_group = match.group(i)
                    if captured_group and re.match(r"^\d+\.?\d*$", captured_group.strip()):
                        value = float(captured_group.strip())
                        break
            if value is not None:
                break
        results[test] = value
        
    return results

def get_normal_range(test_name):
    """Return normal ranges for tests"""
    ranges = {
        "Hemoglobin": (13, 17),       # g/dL for adult males, might vary for females (e.g., 12-15 for females)
        "Blood Sugar": (70, 120),     # mg/dL (fasting glucose typically 70-100, post-meal up to 140 is common threshold)
        "Cholesterol": (0, 200),      # mg/dL (Total Cholesterol, ideal <200)
        "WBC": (4.5, 11),             # K/μL or x10^9/L (4,500 to 11,000 per microliter)
        "Platelets": (150, 450)       # K/μL or x10^9/L (150,000 to 450,000 per microliter)
    }
    return ranges.get(test_name, (0, 0))

if uploaded_file:
    extracted_text = ""
    file_type = uploaded_file.type

    if file_type.startswith('image/'):
        # Original image processing
        image = Image.open(uploaded_file)
        processed_img = preprocess_image(image)
        extracted_text = pytesseract.image_to_string(processed_img)
        st.info("Image text extracted successfully.")
        
    elif file_type == "application/pdf":
        # Original PDF processing
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.info("PDF text extracted successfully.")
        
    elif file_type == "text/csv":
        # Handle CSV files
        try:
            df_csv = pd.read_csv(uploaded_file)
            extracted_text = extract_text_from_tabular(df_csv)
            st.info("CSV data processed into text for extraction.")
            with st.expander("🔍 View Raw CSV Data (first 5 rows)"):
                st.dataframe(df_csv.head())
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            extracted_text = "" # Ensure extracted_text is empty on error
            
    elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        # Handle XLSX files
        try:
            df_excel = pd.read_excel(uploaded_file)
            extracted_text = extract_text_from_tabular(df_excel)
            st.info("Excel data processed into text for extraction.")
            with st.expander("🔍 View Raw Excel Data (first 5 rows)"):
                st.dataframe(df_excel.head())
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            extracted_text = "" # Ensure extracted_text is empty on error
    else:
        st.warning(f"Unsupported file type: {file_type}")
        extracted_text = ""

    st.subheader("Analysis Results")

    if not extracted_text:
        st.error("No text/data could be extracted from the file. Please ensure the file is valid and readable.")
    else:
        with st.expander("🔍 View Extracted Text (for debugging)"):
            st.code(extracted_text, language='text')

        test_results = extract_medical_values(extracted_text)
        
        results = []
        all_values_not_found = True
        for test, value in test_results.items():
            normal_min, normal_max = get_normal_range(test)
            status = "Not Found"
            
            if value is not None:
                all_values_not_found = False
                status = "Normal"
                if value < normal_min:
                    status = "Low"
                elif value > normal_max:
                    status = "High"
            
            results.append({
                "Test": test,
                "Your Value": f"{value:.2f}" if value is not None else "Not Found",
                "Normal Range": f"{normal_min}-{normal_max}",
                "Status": status
            })
        
        df = pd.DataFrame(results)
        
        def color_status(val):
            if val == "High":
                return "color: red; font-weight: bold"
            elif val == "Low":
                return "color: orange; font-weight: bold"
            elif val == "Normal":
                return "color: green"
            else:
                return ""
        
        st.dataframe(
            df.style.applymap(color_status, subset=['Status']),
            height=300,
            use_container_width=True
        )
        
        if all_values_not_found:
            st.info("No specific medical values (Hemoglobin, Blood Sugar, Cholesterol, WBC, Platelets) could be extracted from the report based on the defined patterns. Please ensure the report contains these parameters or adjust the extraction patterns.")

        st.subheader("📈 Value Comparison (Bar Chart)")
        plot_df = df[df['Your Value'] != "Not Found"].copy()
        
        if not plot_df.empty:
            plot_df['Your Value'] = plot_df['Your Value'].astype(float)
            plot_df['Normal Min'] = plot_df['Normal Range'].str.split('-').str[0].astype(float)
            plot_df['Normal Max'] = plot_df['Normal Range'].str.split('-').str[1].astype(float)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(plot_df))
            bar_width = 0.6

            colors = ['red' if s in ['High', 'Low'] else 'green' 
                      for s in plot_df['Status']]
            ax.bar(x, plot_df['Your Value'], width=bar_width, color=colors, label='Your Value')
            
            for j in range(len(plot_df)):
                ax.hlines(y=plot_df['Normal Min'].iloc[j], xmin=x[j] - bar_width/2, xmax=x[j] + bar_width/2,
                          color='blue', linestyle='--', linewidth=2, label='Normal Min' if j == 0 else "")
                ax.hlines(y=plot_df['Normal Max'].iloc[j], xmin=x[j] - bar_width/2, xmax=x[j] + bar_width/2,
                          color='purple', linestyle='--', linewidth=2, label='Normal Max' if j == 0 else "")
                
            for j, row in enumerate(plot_df.iterrows()): # This loop needs to use j for indexing as well
                row_data = row[1] # row[0] is index, row[1] is the Series data
                text_y_position = row_data['Your Value'] + (ax.get_ylim()[1] * 0.02)
                
                ax.text(x[j], text_y_position,
                        f"{row_data['Your Value']:.2f}", 
                        ha='center', va='bottom',
                        color=colors[j], fontsize=9)
            
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df['Test'])
            ax.set_ylabel("Value")
            ax.set_title("Medical Test Results vs. Normal Ranges")
            
            from matplotlib.lines import Line2D
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, fc='green', label='Your Value (Normal)'),
                plt.Rectangle((0, 0), 1, 1, fc='red', label='Your Value (High/Low)'),
                Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Normal Min'),
                Line2D([0], [0], color='purple', lw=2, linestyle='--', label='Normal Max')
            ]
            ax.legend(handles=legend_elements, loc='upper left')

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
        else:
            st.warning("No measurable values found for visualization. Please ensure values are extracted successfully.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Developed with ❤️ by Muhammad Azeem</p>
    <p>Medical Report Analysis System v2.1</p>
</div>
""", unsafe_allow_html=True)