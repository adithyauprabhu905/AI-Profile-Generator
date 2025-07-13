import streamlit as st
import fitz  # PyMuPDF
from word2number import w2n
from fpdf import FPDF
import os
import tempfile
import base64
import re
import pytesseract
from PIL import Image
import numpy as np
import cv2
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import google.generativeai as genai  # Import for Gemini API

# Configure page
st.set_page_config(page_title="AI Profile Generator", layout="wide")

# Function to extract text from PDF with OCR
def extract_text_from_pdf(pdf_path, use_ocr=True):
    doc = fitz.open(pdf_path)
    text = ""
    
    if not use_ocr:
        # Simple text extraction without OCR
        for page in doc:
            text += page.get_text()
        return text
    
    # Use OCR for better extraction
    for page_num, page in enumerate(doc):
        # Get the pixmap of the page
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert PIL image to OpenCV format for preprocessing
        open_cv_image = np.array(img) 
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Convert back to PIL for OCR
        pil_img = Image.fromarray(thresh)
        
        # Extract text using OCR
        page_text = pytesseract.image_to_string(pil_img, lang='eng')
        text += page_text + "\n\n"
    
    return text

def extract_skills_from_certificates(text):
    communication_skills = [
        "communication", "presentation", "public speaking", "negotiation", "interpersonal skills"
    ]
    soft_skills = [
        "leadership", "teamwork", "problem-solving", "critical thinking", "time management",
        "adaptability", "creativity", "emotional intelligence", "conflict resolution", "decision making"
    ]
    technical_skills = [
        "python", "java", "c++", "machine learning", "data analysis", "data science", "ai",
        "artificial intelligence", "web development", "html", "css", "javascript", "sql",
        "cloud computing", "aws", "azure", "devops", "cybersecurity", "networking"
    ]
    business_skills = [
        "project management", "marketing", "sales", "business analysis", "financial modeling",
        "entrepreneurship", "digital marketing", "seo", "content writing", "market research"
    ]
    healthcare_skills = [
        "first aid", "emergency response", "phlebotomy", "basic life support", "nursing", 
        "public health", "clinical research", "medical terminology", "telemedicine", "cpr"
    ]
    education_skills = [
        "classroom management", "curriculum design", "instructional design", "pedagogy",
        "lesson planning", "educational technology", "special education", "student engagement"
    ]
    design_skills = [
        "graphic design", "photoshop", "illustrator", "autocad", "ui design", "ux design",
        "canva", "figma", "adobe xd", "video editing"
    ]
    law_skills = [
        "legal research", "contract drafting", "cyber law", "intellectual property", 
        "human rights", "legal writing", "labor law", "compliance"
    ]

    all_skills = (
        communication_skills + soft_skills + technical_skills +
        business_skills + healthcare_skills + education_skills +
        design_skills + law_skills
    )

    found_skills = []
    text_lower = text.lower()

    for skill in all_skills:
        if skill.lower() in text_lower:
            found_skills.append(skill.title())

    return {
        
        "skills": sorted(set(found_skills)),
        "certificate_names": re.findall(r"(certificate\s+in\s+[A-Za-z\s]+)", text, re.IGNORECASE),
        "organizations": re.findall(r"(?:issued\s+by|provided\s+by|from)\s+([A-Za-z\s&]+)", text, re.IGNORECASE),
        "dates": re.findall(r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", text)
    }


# Function to extract marks from marksheets using OCR and pattern recognition
def extract_marks_from_marksheet(text, board_type):
    subjects = {}
    total = 0

    lines = text.upper().split("\n")
    for line in lines:
        line = line.strip()
        if not line or any(keyword in line for keyword in [
            "ROLL", "NAME", "DOB", "GRADE", "DATE", "SCHOOL", "BOARD", "STATEMENT",
            "CERTIFICATE", "CONTROLLER", "SUB CODE", "TOTAL", "PERCENTAGE", "RESULT",
            "DIVISION", "SIGNATURE", "ISSUE"
        ]):
            continue

        # Extract subject name (text before first number)
        match_subject = re.match(r"([A-Z\s\(\)\+\-\/\.]+?)\s+\d+", line)
        numbers = list(map(int, re.findall(r"\d{1,3}", line)))

        if match_subject and len(numbers) >= 2:
            subject = match_subject.group(1).strip().title()
            max_mark, obtained_mark = numbers[-2], numbers[-1]

            # Heuristic to confirm which number is the actual marks obtained
            if 0 <= obtained_mark <= max_mark <= 150:
                subjects[subject] = obtained_mark
            else:
                # fallback in case the above fails
                subjects[subject] = numbers[-1]

    total = sum(subjects.values())
    max_total = len(subjects) * 100 if board_type in ["CBSE", "STATE"] else len(subjects) * 150
    percentage = (total / max_total) * 100 if max_total else 0

    return {"subjects": subjects, "total": total, "percentage": round(percentage, 2)}


# UPDATED FUNCTION: Generate recommendations using Gemini API with direct API key
def generate_gemini_recommendations(name, sslc_data, cbse_data, skills):
    # Using the API key directly in the code
    api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Build prompt with student data
        prompt = f"Based on the following student information, provide personalized career and academic recommendations:\n\n"
        prompt += f"Student Name: {name}\n\n"
        
        if sslc_data:
            prompt += "SSLC (10th Grade) Results:\n"
            prompt += f"- Overall Percentage: {sslc_data.get('percentage', 0):.2f}%\n"
            if "subjects" in sslc_data and sslc_data["subjects"]:
                prompt += "- Subject-wise Marks:\n"
                for subject, mark in sslc_data["subjects"].items():
                    prompt += f"  * {subject}: {mark}\n"
                
        if cbse_data:
            prompt += "\nCBSE (12th Grade) Results:\n"
            prompt += f"- Overall Percentage: {cbse_data.get('percentage', 0):.2f}%\n"
            if "subjects" in cbse_data and cbse_data["subjects"]:
                prompt += "- Subject-wise Marks:\n"
                for subject, mark in cbse_data["subjects"].items():
                    prompt += f"  * {subject}: {mark}\n"
                    
        if skills:
            prompt += "\nSkills Identified from Certificates:\n"
            for skill in skills:
                prompt += f"- {skill}\n"
        
        prompt += "\nBased on this information, please provide:\n"
        prompt += "1. Career path recommendations (2-3 options)\n"
        prompt += "2. Suggested areas for skill development\n"
        prompt += "3. Educational recommendations (courses, degrees)\n"
        prompt += "4. Strengths based on the academic performance\n"
        prompt += "5. Areas of improvement\n"
        prompt += "\nPlease format the response in clear sections with bullet points where appropriate."
        
        # Use Gemini-Pro model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate response from Gemini
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
            )
        )
        
        # Extract recommendation text from the response
        recommendation = response.text
        return recommendation
    
    except Exception as e:
        # Provide more useful error message
        st.warning(f"Error using Gemini API: {str(e)}. Falling back to basic recommendations.")
        return generate_llm_recommendations_fallback(name, sslc_data, cbse_data, skills)


# Fallback LLM recommendations when Gemini API is not available
def generate_llm_recommendations_fallback(name, sslc_data, cbse_data, skills):
    # Create a basic recommendation based on the data
    recommendation = "# AI-Generated Recommendations\n\n"
    
    # Analyze academic performance
    high_performance = False
    science_focus = False
    arts_focus = False
    commerce_focus = False
    
    if sslc_data and sslc_data.get("percentage", 0) > 80:
        high_performance = True
    if cbse_data and cbse_data.get("percentage", 0) > 80:
        high_performance = True
        
    # Check for subject focus in 12th grade
    if cbse_data and "subjects" in cbse_data:
        subjects = [s.lower() for s in cbse_data["subjects"].keys()]
        if any(s in subjects for s in ["physics", "chemistry", "biology", "mathematics"]):
            science_focus = True
        if any(s in subjects for s in ["economics", "business", "accountancy", "commerce"]):
            commerce_focus = True
        if any(s in subjects for s in ["history", "geography", "political science", "sociology", "literature"]):
            arts_focus = True
    
    # Career path recommendations
    recommendation += "## Career Path Recommendations\n\n"
    
    if science_focus:
        if high_performance:
            recommendation += "* **Engineering or Medical Sciences**: Your strong performance in science subjects suggests you would excel in these fields.\n"
            recommendation += "* **Research & Development**: Consider research-oriented careers in pure sciences or applied technologies.\n"
        else:
            recommendation += "* **Applied Sciences**: Fields like laboratory technology, paramedical sciences, or environmental science.\n"
            recommendation += "* **Technical Support Roles**: Positions where scientific knowledge is applied in practical settings.\n"
    elif commerce_focus:
        recommendation += "* **Business Management**: Your commerce background prepares you for business administration roles.\n"
        recommendation += "* **Financial Services**: Consider careers in banking, investment analysis, or accounting.\n"
    elif arts_focus:
        recommendation += "* **Content Creation**: Your humanities background is well-suited for content writing, journalism, or media.\n"
        recommendation += "* **Social Services**: Consider careers in education, counseling, or public administration.\n"
    else:
        recommendation += "* **Interdisciplinary Fields**: Consider fields that combine multiple areas of knowledge like digital marketing or educational technology.\n"
        recommendation += "* **Entrepreneurship**: Your diverse knowledge base could be well-suited for starting your own venture.\n"
    
    # Skill development suggestions
    recommendation += "\n## Suggested Areas for Skill Development\n\n"
    current_skills = [s.lower() for s in skills]
    
    if not any(s in current_skills for s in ["communication", "public speaking", "presentation"]):
        recommendation += "* **Communication Skills**: Essential in any career path and will complement your academic background.\n"
    
    if not any(s in current_skills for s in ["leadership", "teamwork", "management"]):
        recommendation += "* **Leadership & Teamwork**: Developing these skills will enhance your employability across sectors.\n"
    
    if not any(s in current_skills for s in ["python", "data analysis", "programming", "coding"]):
        recommendation += "* **Digital Literacy & Programming**: Basic programming knowledge is increasingly valuable in most fields.\n"
        
    # Educational recommendations
    recommendation += "\n## Educational Recommendations\n\n"
    
    if science_focus:
        if high_performance:
            recommendation += "* **Bachelor's Degree**: B.Tech, B.Sc, or MBBS depending on your specific interests.\n"
            recommendation += "* **Consider**: IITs, NITs, AIIMS or other premier institutions if your scores permit.\n"
        else:
            recommendation += "* **Bachelor's Degree**: B.Sc in applied sciences or B.Tech in emerging fields.\n"
            recommendation += "* **Consider**: Specialized diploma courses alongside your degree for practical skills.\n"
    elif commerce_focus:
        recommendation += "* **Bachelor's Degree**: BBA, B.Com, or BA Economics.\n"
        recommendation += "* **Professional Certifications**: Consider CA, CS, or CFA depending on your interests.\n"
    elif arts_focus:
        recommendation += "* **Bachelor's Degree**: BA in your subject of interest or integrated courses.\n"
        recommendation += "* **Consider**: Certificate courses in digital skills to complement your humanities background.\n"
    else:
        recommendation += "* **Interdisciplinary Programs**: Look for programs that allow flexibility in course selection.\n"
        recommendation += "* **Online Learning**: Supplement formal education with online courses in high-demand skills.\n"
    
    # Strengths
    recommendation += "\n## Strengths Based on Academic Performance\n\n"
    
    if high_performance:
        recommendation += "* **Consistent Academic Excellence**: Your high scores demonstrate strong learning ability and discipline.\n"
        recommendation += "* **Subject Mastery**: You've shown particularly strong understanding in your core subjects.\n"
    else:
        recommendation += "* **Balanced Performance**: You've demonstrated capability across different subject areas.\n"
    
    if cbse_data and "subjects" in cbse_data:
        best_subjects = sorted(cbse_data["subjects"].items(), key=lambda x: x[1], reverse=True)[:2]
        if best_subjects:
            recommendation += f"* **Subject Strengths**: Particular aptitude in {best_subjects[0][0]}"
            if len(best_subjects) > 1:
                recommendation += f" and {best_subjects[1][0]}"
            recommendation += ".\n"
    
    # Areas of improvement
    recommendation += "\n## Areas of Improvement\n\n"
    
    if cbse_data and "subjects" in cbse_data:
        worst_subjects = sorted(cbse_data["subjects"].items(), key=lambda x: x[1])[:1]
        if worst_subjects:
            recommendation += f"* **Academic Focus**: Consider additional attention to {worst_subjects[0][0]}.\n"
    
    if not skills:
        recommendation += "* **Practical Skills**: Focus on developing practical and certification-backed skills.\n"
    
    recommendation += "* **Continuous Learning**: Develop a habit of self-learning beyond curriculum requirements.\n"
    recommendation += "* **Project-Based Experience**: Seek opportunities for hands-on projects to apply theoretical knowledge.\n"
    
    return recommendation


# Function to generate a summary based on performance and skills
def generate_summary(sslc_data, cbse_data, skills):
    summary = "Academic Summary:\n\n"

    if sslc_data:
        summary += f"- SSLC (10th Grade) Percentage: {sslc_data.get('percentage', 0):.2f}%\n"
        summary += f"- Total Marks (SSLC): {sslc_data.get('total', 0)}\n"
        if sslc_data.get("subjects"):
            summary += "- Subjects (SSLC):\n"
            for subject, mark in sslc_data["subjects"].items():
                summary += f"  - {subject}: {mark}\n"
            summary += "\n"

    if cbse_data:
        summary += f"- CBSE (12th Grade) Percentage: {cbse_data.get('percentage', 0):.2f}%\n"
        summary += f"- Total Marks (CBSE): {cbse_data.get('total', 0)}\n"
        if cbse_data.get("subjects"):
            summary += "- Subjects (CBSE):\n"
            for subject, mark in cbse_data["subjects"].items():
                summary += f"  - {subject}: {mark}\n"
            summary += "\n"

    if skills:
        summary += "Skills Identified from Certificates:\n"
        for skill in skills:
            summary += f"- {skill}\n"
    else:
        summary += "No specific skills identified from certificates.\n"

    return summary


# Function to create an AI profile PDF
def create_ai_profile(name, sslc_data, cbse_data, skills, recommendations):
    pdf = FPDF()
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(190, 10, "Student AI Profile", 0, 1, "C")
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    # Add name
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "Name:", 0, 0)
    pdf.set_font("Arial", "", 12)
    pdf.cell(150, 10, name, 0, 1)
    pdf.ln(5)
    
    # Add SSLC data if available
    if sslc_data:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(190, 10, "SSLC (10th Grade) Results", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(40, 10, "Percentage:", 0, 0)
        pdf.cell(150, 10, f"{sslc_data.get('percentage', 0):.2f}%", 0, 1)
        
        if "subjects" in sslc_data and sslc_data["subjects"]:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(190, 10, "Subject-wise Marks:", 0, 1)
            pdf.set_font("Arial", "", 12)
            for subject, marks in sslc_data["subjects"].items():
                pdf.cell(100, 8, subject, 0, 0)
                pdf.cell(90, 8, str(marks), 0, 1)
        pdf.ln(5)
    
    # Add CBSE data if available
    if cbse_data:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(190, 10, "CBSE (12th Grade) Results", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(40, 10, "Percentage:", 0, 0)
        pdf.cell(150, 10, f"{cbse_data.get('percentage', 0):.2f}%", 0, 1)
        
        if "subjects" in cbse_data and cbse_data["subjects"]:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(190, 10, "Subject-wise Marks:", 0, 1)
            pdf.set_font("Arial", "", 12)
            for subject, marks in cbse_data["subjects"].items():
                pdf.cell(100, 8, subject, 0, 0)
                pdf.cell(90, 8, str(marks), 0, 1)
        pdf.ln(5)
    
    # Add skills
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "Skills Identified from Certificates:", 0, 1)
    pdf.set_font("Arial", "", 12)
    if skills:
        for skill in skills:
            pdf.cell(190, 8, f"- {skill}", 0, 1)
    else:
        pdf.cell(190, 8, "No specific skills identified", 0, 1)
    pdf.ln(5)
    
    # Add recommendations
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "AI Recommendations:", 0, 1)
    pdf.set_font("Arial", "", 12)
    
    # Handle multi-line recommendations
    lines = recommendations.split('\n')
    for line in lines:
        pdf.multi_cell(190, 8, line)
    
    # Save PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp_file.name
    pdf.output(pdf_path)
    return pdf_path

# Function to display download link
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}" target="_blank">{file_label}</a>'

# Main Streamlit app
def main():
    st.title("Enhanced AI Profile Generator")
    st.write("Upload your marksheets and certificates to generate a comprehensive AI profile with career recommendations.")

    # ✅ Initialize session state
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    if 'marksheets_data' not in st.session_state:
        st.session_state.marksheets_data = {}
    if 'certificates_data' not in st.session_state:
        st.session_state.certificates_data = {}
    if 'skills' not in st.session_state:
        st.session_state.skills = []
    if 'certificate_count' not in st.session_state:
        st.session_state.certificate_count = 0
    if 'debug_ocr_text' not in st.session_state:
        st.session_state.debug_ocr_text = ""
    if 'sslc_data' not in st.session_state:
        st.session_state.sslc_data = {}
    if 'cbse_data' not in st.session_state:
        st.session_state.cbse_data = {}

    # Student name input
    student_name = st.text_input("Student Name")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Marksheets", "Certificates", "Summary & Generate"])

    # MARKSHEET TAB
    with tab1:
        st.subheader("Upload Marksheets")
        board_type = st.radio("Select Board Type", ["CBSE", "State Board"])
        grade_type = st.selectbox("Select Grade", ["10th Grade", "12th Grade", "Graduation/Degree"])
        marksheet_file = st.file_uploader("Upload Marksheet PDF", type=['pdf', 'jpg', 'jpeg', 'png'], key="marksheet_uploader")
        use_ocr = st.checkbox("Use OCR for better extraction (slower but more accurate)", value=True)

        if marksheet_file is not None:
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, marksheet_file.name)
            with open(temp_path, "wb") as f:
                f.write(marksheet_file.getbuffer())

            if st.button("Process Marksheet"):
                with st.spinner("Processing marksheet with OCR..."):
                    try:
                        text = extract_text_from_pdf(temp_path, use_ocr=use_ocr)
                        extracted_data = extract_marks_from_marksheet(text, board_type)

                        # ✅ Save and assign based on grade
                        key = f"{board_type}_{grade_type}"
                        st.session_state.marksheets_data[key] = extracted_data
                        st.session_state.processed_files[marksheet_file.name] = f"{board_type} {grade_type} Marksheet"

                        if "10th" in grade_type:
                            st.session_state.sslc_data = extracted_data
                        elif "12th" in grade_type:
                            st.session_state.cbse_data = extracted_data

                        if extracted_data["subjects"]:
                            st.success(f"Marksheet processed! Extracted {len(extracted_data['subjects'])} subjects.")
                            st.write(f"Percentage: {extracted_data['percentage']:.2f}%")
                            st.write(f"Total Marks: {extracted_data['total']}")
                            subjects_df = pd.DataFrame(list(extracted_data["subjects"].items()), columns=["Subject", "Marks"])
                            st.write("Subject-wise Marks:")
                            st.dataframe(subjects_df)
                        else:
                            st.warning("Processed the marksheet but couldn't identify subjects. Try enabling OCR if not already enabled.")
                            with st.expander("Show OCR Text (for debugging)"):
                                st.text(text)

                    except Exception as e:
                        st.error(f"Error processing marksheet: {str(e)}")
                        with st.expander("Show OCR Text (for debugging)"):
                            st.text(st.session_state.debug_ocr_text)

    # CERTIFICATE TAB
    with tab2:
        st.subheader("Upload Certificates")
        st.session_state.certificate_count = st.number_input("Enter number of certificates:", min_value=0, step=1)
        certificate_file = st.file_uploader("Upload Certificate PDF", type=['pdf', 'jpg', 'jpeg', 'png'], key="certificate_uploader")

        if certificate_file is not None:
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, certificate_file.name)
            with open(temp_path, "wb") as f:
                f.write(certificate_file.getbuffer())

            if st.button("Process Certificate"):
                with st.spinner("Processing certificate with OCR..."):
                    try:
                        text = extract_text_from_pdf(temp_path, use_ocr=use_ocr)
                        certificate_data = extract_skills_from_certificates(text)
                        cert_key = f"certificate_{st.session_state.certificate_count}"
                        st.session_state.certificate_count += 1
                        st.session_state.certificates_data[cert_key] = certificate_data
                        st.session_state.processed_files[certificate_file.name] = "Certificate"

                        if certificate_data["skills"]:
                            st.session_state.skills = list(set(st.session_state.skills + certificate_data["skills"]))

                        st.success("Certificate processed successfully!")

                        if certificate_data["certificate_names"]:
                            st.write("Certificate/Course Names:")
                            for cert_name in certificate_data["certificate_names"]:
                                st.write(f"- {cert_name}")
                        if certificate_data["organizations"]:
                            st.write("Issuing Organizations:")
                            for org in certificate_data["organizations"]:
                                st.write(f"- {org}")
                        if certificate_data["dates"]:
                            st.write("Dates:")
                            for date in certificate_data["dates"]:
                                st.write(f"- {date}")
                        if certificate_data["skills"]:
                            st.write("Skills identified:")
                            for skill in certificate_data["skills"]:
                                st.write(f"- {skill}")
                        else:
                            st.info("No specific skills were identified in this certificate.")

                        with st.expander("Show OCR Text (for debugging)"):
                            st.text(text)

                    except Exception as e:
                        st.error(f"Error processing certificate: {str(e)}")
    with tab3:
        st.subheader("Summary & Generate AI Profile")

        if st.session_state.processed_files:
            st.write("### Processed Files")
            for filename, file_type in st.session_state.processed_files.items():
                st.write(f"✅ {filename} - {file_type}")

        # AI Recommendation Option
        recommendation_option = st.radio(
            "Choose Recommendation Source:",
            ["Basic Summary", "AI-Powered Recommendations"]
        )

        st.subheader("Generate AI Profile")
        if st.button("Generate AI Profile"):
            if not student_name:
                st.warning("Please enter student name before generating profile.")
            elif not st.session_state.sslc_data and not st.session_state.cbse_data:
                st.warning("Please process at least one marksheet before generating profile.")
            else:
                with st.spinner("Generating AI Profile..."):
                    try:
                        # Generate recommendations based on selected option
                        if recommendation_option == "Basic Summary":
                            recommendations = generate_summary(st.session_state.sslc_data, st.session_state.cbse_data, st.session_state.skills)
                        else:
                            recommendations = generate_gemini_recommendations(
                                student_name,
                                st.session_state.sslc_data,
                                st.session_state.cbse_data,
                                st.session_state.skills
                            )
                        
                        pdf_path = create_ai_profile(
                            student_name,
                            st.session_state.sslc_data,
                            st.session_state.cbse_data,
                            st.session_state.skills,
                            recommendations
                        )
                        st.success("AI Profile generated successfully!")
                        st.markdown(get_binary_file_downloader_html(pdf_path, 'Download AI Profile PDF'), unsafe_allow_html=True)
                        st.subheader("AI Recommendations Preview")
                        st.write(recommendations)

                    except Exception as e:
                        st.error(f"Error generating profile: {str(e)}")
if __name__ == "__main__":
    main()
