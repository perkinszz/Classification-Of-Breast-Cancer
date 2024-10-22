import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import cv2
import pytesseract
import time  # For simulating processing time
import numpy as np
import pytesseract
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Load the dataset (update the path as necessary)
data = pd.read_csv('/Users/mac/Downloads/Breast_Cancer_METABRIC_modified1.csv')


# Set the page configuration
st.set_page_config(page_title="Classification Of Breast Cancer", layout="wide")




# Background image URLs
default_background_image_url = "https://images.unsplash.com/photo-1581594624138-defffbdc0f0f?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
login_background_image_url = "https://cdn.pixabay.com/photo/2020/03/30/09/15/corona-4983590_1280.jpg"
donate_background_image_url = "https://images.unsplash.com/photo-1631879742033-df44819e6a52?q=80&w=2532&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
awareness_background_image_url = "https://images.unsplash.com/photo-1678795014601-f30ab980cfb9?q=80&w=2574&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
cancer_background_image_url= "https://images.unsplash.com/photo-1612832164313-ac0d7e07b5ce?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
Chat_background_image_url = "https://cdn.pixabay.com/photo/2014/03/29/23/49/the-background-301145_1280.png"
upload_background_image_url ="https://images.unsplash.com/photo-1620120966883-d977b57a96ec?q=80&w=2532&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
# Function to set background image

def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



    # Function to load and preprocess the image
def load_image(image_file):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    return img

# Function to extract text from image using Tesseract OCR
def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# Function to mock prediction (replace with your trained model)
def mock_predict(text):
    if "cancer" in text.lower():
        return "Positive"
    else:
        return "Negative"

# Navigation sidebar
st.sidebar.title("G-one Medik")
options = st.sidebar.radio("Welcome to G-Medik:", 
                           ["Home", "Patient Login", "Donate", "Awareness", 
                            "Cancer Detection", "Chat with our agent", "MRI Scan"])


# Set background based on selected page
if options == "Patient Login":
    set_background_image(login_background_image_url)
elif options == "Donate":
    set_background_image(donate_background_image_url)  # Set specific background for donate page
else:
    set_background_image(default_background_image_url)

# Header
# Conditional Header Rendering
if options not in ["Cancer Detection"]:
 if options not in ["Upload MRI Scan"]:
  if options not in ["Chat with our agent", "Patient Login", "Donate"]:
    st.title("Welcome to Our Awareness Campaign")
 st.header("Join Us in the Fight Against Breast Cancer")

# Home Page Content
if options == "Home":
    st.subheader("Classification Of Breast Cancer")
    st.write("This platform is dedicated to raising awareness about breast cancer, providing resources, and supporting patients and families.")
   


 
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="get-involved">', unsafe_allow_html=True)
        st.image("https://cdn.pixabay.com/photo/2017/02/01/10/52/comic-characters-2029609_1280.png", width=100)  # Volunteer icon
        st.write("**Volunteer**")
        st.write("Become a volunteer in our mission to combat cancer. Your time and effort can make a significant difference in the lives of patients and their families. By volunteering, you will help raise awareness, support fundraising events, and provide essential resources to those affected by this disease.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="get-involved">', unsafe_allow_html=True)
        st.image("https://cdn.pixabay.com/photo/2019/07/02/05/52/money-4311572_1280.png", width=100)  # Fundraise icon
        st.write("**Fundraise**")
        st.write("We invite you to join our fundraising efforts to support breast cancer awareness and research. Your contributions will provide essential resources for patients and their families, fund innovative research, and promote education on early detection and treatment options. Every dollar counts, whether through organizing events, donating, or spreading the word.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="get-involved">', unsafe_allow_html=True)
        st.image("https://cdn.pixabay.com/photo/2020/08/09/10/23/heart-5475102_1280.png", width=100)  # Partner icon
        st.write("**Partner**")
        st.write("We invite you to partner with us in the fight against breast cancer. Your involvement can significantly raise awareness, support research, and provide essential resources for patients and their families. ")
        st.markdown('</div>', unsafe_allow_html=True)



# Patient Login Page
elif options == "Patient Login":
    set_background_image(login_background_image_url)  # Set specific background for login page
    
    st.title("Patient Login")
    
     # Input for Patient ID
    patient_id = st.text_input("Enter your Patient ID:")
    
    if st.button("Login"):
        # Simulate processing time 
        # Check if the Patient ID exists in the dataset
        if patient_id in data['Patient ID'].astype(str).values:
            # Retrieve patient information
            patient_info = data[data['Patient ID'].astype(str) == patient_id].iloc[0]
            st.success("Login successful!")
            st.write(f"Welcome, Patient ID: {patient_info['Patient ID']}")

            # Set logged-in state
            st.session_state.logged_in = True
            
            # Display patient information in a styled box
            st.subheader("Patient Information")
            st.image("https://img.freepik.com/premium-photo/cancer-awareness-month-breast-cancer-women-power_919955-36820.jpg", width=130,
                      caption="Patient's Passport Photo")

            # Display patient information with formatting
            st.write(f"**Age**: {patient_info['Age at Diagnosis']:.2f} years")
            st.write(f"**Type of Breast Surgery**: {patient_info['Type of Breast Surgery']}")
            st.write(f"**Cancer Type**: {patient_info['Cancer Type']}")
            st.write(f"**Cancer Type Detailed**: {patient_info['Cancer Type Detailed']}")

            # Allow users to download their breast cancer scan after logging in
            csv_file_url = "https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/8830672/e1f73fe0-0abe-49f6-a809-0e8dfb6aab20/Breast_Cancer_METABRIC.csv"
            if st.button('Download Scan Result'):
                # Provide a link to download directly.
                st.markdown(f"[Click here to download your scan]( {csv_file_url} )")

            # Logout button
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.success("You have been logged out.")
                st.experimental_rerun()  # Refresh the app to reset state

        else:
            st.error("Invalid Patient ID. Please try again.")

# Check if user is logged in for other options
    if 'logged_in' in st.session_state and st.session_state.logged_in:
    # Show additional content or options for logged-in users
     pass  # You can add more features for logged-in users here

    else:
        st.sidebar.write("Please log in to access more features.")



# Donate Page with Spinner Animation on Submit
elif options == "Donate":
    set_background_image(donate_background_image_url)  # Set specific background for donate page
  
    st.markdown(
        """
        <h1 style="text-align: left; font-size: 3em; text-shadow: 2px 2px 5px #1C6468;">Make a donation today</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3 style="text-align: left; font-size: 1.5em; text-shadow:2px 1px 5px #424242;">
        Every donation brings us one step closer to a world without breast cancer. Together, we can make a difference in the lives of those affected and support the fight for a cure."
        </h3>
        """,
        unsafe_allow_html=True
    )
    donation_amount = st.number_input("Enter donation amount", min_value=1)

    # Payment Method Selection with Icons
    payment_method = st.selectbox("Select Payment Method:", ["Credit/Debit Card", "PayPal", "Bank Transfer"])

    if payment_method == "Credit/Debit Card":
        st.markdown('<img src="https://cdn.pixabay.com/photo/2017/02/27/15/39/cheque-guarantee-card-2103509_1280.png" width="50" style="vertical-align: middle;" alt="Visa"> Credit/Debit Card', unsafe_allow_html=True)
        card_number = st.text_input("Card Number", type="password")
        card_expiry = st.text_input("Expiry Date (MM/YY)")
        card_cvc = st.text_input("CVC", type="password")

    elif payment_method == "PayPal":
        st.markdown('<img src="https://cdn.pixabay.com/photo/2018/05/08/21/29/paypal-3384015_1280.png" width="50" style="vertical-align: middle;" alt="PayPal"> PayPal', unsafe_allow_html=True)
        paypal_email = st.text_input("PayPal Email")

    elif payment_method == "Bank Transfer":
        st.markdown('<img src="https://cdn.pixabay.com/photo/2015/10/14/18/43/bank-988164_1280.png" width="50" style="vertical-align: middle;" alt="Bank Transfer"> Bank Transfer', unsafe_allow_html=True)
        bank_account_number = st.text_input("Bank Account Number")
        bank_routing_number = st.text_input("Routing Number")
   
 
    if st.button("Donate"):
        with st.spinner("Processing... Please wait."):
            time.sleep(5)  # Simulate processing time (replace with actual processing logic)

    
        if donation_amount > 0:
            if payment_method == "Credit/Debit Card":
                if card_number and card_expiry and card_cvc:
                    st.success(f"Thank you for your donation of ${donation_amount} via {payment_method}!")
                else:
                    st.error("Please fill in all credit/debit card details.")
            elif payment_method == "PayPal":
                if paypal_email:
                    st.success(f"Thank you for your donation of ${donation_amount} via {payment_method}!")
                else:
                    st.error("Please enter your PayPal email.")
            elif payment_method == "Bank Transfer":
                if bank_account_number and bank_routing_number:
                    st.success(f"Thank you for your donation of ${donation_amount} via {payment_method}!")
                else:
                    st.error("Please fill in all bank transfer details.")
        else:
            st.error("Please enter a valid donation amount.")

# Awareness Page - New section added to navigation
elif options == "Awareness":
    set_background_image(awareness_background_image_url)
    
    # Section: What is Breast Cancer?
    st.subheader("What is Breast Cancer?")
    st.write("""
    Breast cancer is a type of cancer that forms in the cells of the breasts. It can occur in both men and women but is far more common in women. 
    The disease can begin in different parts of the breast and can be invasive or non-invasive.
    """)

   # Section: Causes
    st.subheader("Causes of Breast Cancer")
    st.write("""
   While the exact cause of breast cancer is not fully understood, several risk factors have been identified:
   - **Genetic Factors**: Mutations in genes such as BRCA1 and BRCA2 increase the risk.
   - **Age**: The risk increases with age.
   - **Family History**: A family history of breast cancer can increase risk.
   - **Lifestyle Factors**: Obesity, alcohol consumption, and lack of physical activity are linked to higher breast cancer risk.
   """)

   # Section: Prevention Strategies
    st.subheader("Prevention Strategies")
    st.write("""
   There are several strategies to reduce the risk of developing breast cancer:
   - **Regular Screening**: Mammograms can help detect breast cancer early.
   - **Healthy Lifestyle**: Maintaining a healthy weight, exercising regularly, and limiting alcohol intake can reduce risk.
   - **Genetic Testing**: Individuals with a family history may consider genetic testing for BRCA mutations.
   - **Medications**: For those at high risk, medications like tamoxifen may be recommended.
   """)

   # Section: Treatment Options
    st.subheader("Treatment Options")
    st.write("""
   Treatment for breast cancer may involve a combination of therapies:
   - **Surgery**: Options include lumpectomy (removing the tumor) or mastectomy (removing one or both breasts).
   - **Radiation Therapy**: Often used after surgery to eliminate remaining cancer cells.
   - **Chemotherapy**: Uses drugs to kill cancer cells and is often used before or after surgery.
   - **Hormonal Therapy**: For cancers that are hormone receptor-positive.
   - **Targeted Therapy**: Drugs that specifically target cancer cell functions.
   """)

   # Section: Common Procedures
    st.subheader("Common Procedures")
    st.write("""
   Some common procedures related to breast cancer treatment include:
   - **Biopsy**: A procedure to remove a small sample of tissue for testing.
   - **Mastectomy**: Surgery to remove one or both breasts.
   - **Lumpectomy**: Surgery to remove the tumor and some surrounding tissue.
   - **Sentinel Node Biopsy**: A procedure to check if cancer has spread to nearby lymph nodes.
   """)

# Cancer Detection Page (Ensured no overlap with other pages)
elif options == "Cancer Detection":
    set_background_image(cancer_background_image_url)
    
    # Set up title and subheader for the app
    st.markdown(
        """
        <h1 style="text-align: left; font-size: 4em; text-shadow: 2px 2px 5px #1C6468;">Cancer Detection App</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3 style="text-align: left; font-size: 2.5em; text-shadow:2px 1px 5px #424242;">
        Upload your medical documentation below for a thorough analysis. 
        Early detection can save lives!
        </h3>
        """,
        unsafe_allow_html=True
    )

    # File uploader for image input
    uploaded_file = st.file_uploader("Upload a medical report image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the image
        image = load_image(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Extract text from the uploaded image
        extracted_text = extract_text_from_image(image)
        
        st.write("Extracted Text:")
        st.text_area("Text from Image", extracted_text, height=150)

        # Show a spinner while processing the image
        if st.button("Cancer Status"):
            with st.spinner("Processing... Please wait."):
                time.sleep(5)  # Simulate processing time (replace with actual processing logic)
                prediction = mock_predict(extracted_text)
                st.success(f"Cancer Status: {prediction}")

                #MRI Scan Section
elif options == "MRI Scan":
    set_background_image(upload_background_image_url)

     # Set up title and subheader for the app
    st.markdown(
        """
        <h1 style="text-align: left; font-size: 4em; text-shadow: 2px 2px 5px #1C6468;">Upload Your MRI Scan</h1>
        """,
        unsafe_allow_html=True
    )
 

    uploaded_file = st.file_uploader("Choose an MRI scan file...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded MRI Scan', use_column_width=True)

        # Placeholder for model prediction logic (replace with your model's prediction function)
        if st.button("Analyze"):
            # Show a spinner while processing the image
            with st.spinner("Processing... Please wait."):
                time.sleep(5)  # Simulate processing time (replace with actual processing logic) 
            # Simulated result (replace this with actual model prediction logic)
            result = "The MRI scan is healthy."  # Replace with your model's output
            
            st.success(result)


# Chat with Agent Page Content
if options == "Chat with our agent":
    st.markdown(
        """
        <h1 style="text-align: left; font-size: 4em; text-shadow: 2px 2px 5px #1C6468;">Chat with Alita</h1>
        """,
        unsafe_allow_html=True
    )

    set_background_image(Chat_background_image_url)
    # Embed the Dialogflow chatbot using iframe
    st.components.v1.html("""
    <iframe allow="microphone;" height="500" width="850" src="https://console.dialogflow.com/api-client/demo/embedded/675901f4-33f0-4d8e-86b9-54b80c0c41dd"></iframe>
    """, height=500)  # Adjust height if necessary for visibility

# Additional pages can be added similarly...

# Footer with solid black border
st.markdown('<div class="footer" style="border-top:3px solid white; padding-top:10px; padding-bottom:10px;">', unsafe_allow_html=True)
st.markdown("### Contact Us")
st.write("For more information, visit our social media:")
st.sidebar.markdown("---")
st.sidebar.write("© 2024")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("https://cdn.pixabay.com/photo/2017/06/22/06/22/facebook-2429746_1280.png", width=20)  # Facebook icon
    st.write("[Facebook](https://www.facebook.com)")

with col2:
    st.image("https://cdn.pixabay.com/photo/2017/06/22/14/23/twitter-2430933_1280.png", width=20)  # Twitter icon
    st.write("[Twitter](https://www.twitter.com)")

with col3:
    st.image("https://cdn.pixabay.com/photo/2022/04/01/05/40/app-7104075_1280.png", width=20)  # Instagram icon
    st.write("[Instagram](https://www.instagram.com)")

with col4:
    st.image("https://cdn.pixabay.com/photo/2017/08/22/11/56/linked-in-2668700_1280.png", width=20)  # LinkedIn icon
    st.write("[LinkedIn](https://www.linkedin.com)")
    st.markdown('</div>', unsafe_allow_html=True)