###############################
# app.py (Single-File App)
###############################

import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import folium_static
from datetime import datetime
import xmltodict
import time

# For AI Chatbot
# Make sure to install: pip install transformers
from transformers import pipeline

# Set up a local Question-Answering pipeline (distilbert-based)
# This model is small enough to run locally without API keys.
@st.cache_resource
def load_qa_pipeline():
    try:
        nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        return nlp
    except:
        return None

nlp_qa = load_qa_pipeline()

###############################
# PAGE CONFIG & GLOBAL STYLING
###############################
# Hide Streamlit's default menu and footer
st.set_page_config(
    page_title="Cancer Support App",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS to remove default Streamlit elements and apply a dark theme
custom_css = """
<style>
/* Hide the default Streamlit elements (hamburger menu, footer, header) */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Global body styling */
body, .css-18e3th9, .css-1cpxqw2, .css-12oz5g7 {
    background: #000000 !important;
    color: #FFFFFF !important;
    font-family: "Arial", sans-serif;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #333333;
}
::-webkit-scrollbar-thumb {
    background-color: #888888;
    border-radius: 10px;
}

/* Navbar container */
.navbar {
    width: 100%;
    background-color: #111111;
    display: flex;
    justify-content: center;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 999;
    border-bottom: 1px solid #444;
}

/* Navbar links */
.navbar a {
    color: #FFFFFF;
    padding: 16px 20px;
    text-decoration: none;
    text-transform: uppercase;
    margin: 0 10px;
    font-weight: 600;
    transition: 0.3s;
    border-bottom: 2px solid transparent;
    letter-spacing: 0.5px;
}

.navbar a:hover {
    border-bottom: 2px solid #FFFFFF;
    color: #00FFFF;
}

/* Hero section style (Home Page) */
.hero-section {
    padding: 100px 20px;
    text-align: center;
    background: linear-gradient(135deg, #050505 0%, #141414 100%);
    border-bottom: 1px solid #333;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 20px;
    color: #FFFFFF;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #CCCCCC;
    max-width: 800px;
    margin: 0 auto 40px auto;
}

.hero-image {
    max-width: 600px;
    width: 100%;
    border: 2px solid #444;
    border-radius: 10px;
}

/* Section headers */
section h2 {
    font-size: 2rem;
    margin-top: 40px;
    margin-bottom: 20px;
    font-weight: 700;
    color: #FFFFFF;
    text-align: center;
}

/* Card-like container for sub-sections */
.resource-card {
    background-color: #1a1a1a;
    padding: 20px;
    margin: 10px 0;
    border-radius: 8px;
    border: 1px solid #333;
}

/* Buttons (for forms, etc.) */
.stButton>button {
    background-color: #444444;
    color: #FFFFFF;
    border: none;
    padding: 10px 20px;
    margin: 10px 0;
    transition: 0.3s;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #00FFFF;
    color: #000000;
    cursor: pointer;
}

/* Dataframe styling */
.css-1nh1xq3, .css-1x1pioo, .css-2ykyy6, .css-0, .stDataFrame {
    background-color: #222222 !important;
    color: #FFFFFF !important;
}

/* Additional minor tweaks */
h1, h2, h3, h4, h5, h6, p, div, span, label {
    color: #FFFFFF !important;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


###############################
# NAVIGATION LOGIC
###############################
# We store the current page in session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# List of pages
PAGES = [
    "Home",
    "Locate Hospitals",
    "Accommodation Resources",
    "Latest Research",
    "Financial Support",
    "Clinical Trials",
    "Emotional & Social Support",
    "Interactive Tools & Extras",
    "Symptom Checker",
    "Medication Tracker",
    "Personal Journal",
    "Appointment Scheduler",
    "AI Chatbot (Beta)"
]

def navbar():
    """Render a custom top navbar with clickable links to switch pages."""
    st.markdown(
        """
        <div class="navbar">
        """ +
        "".join([f'<a href="?page={page.replace(" ", "+")}">{page}</a>' for page in PAGES]) +
        """
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Read query params to see if user switched pages
    query_params = st.experimental_get_query_params()
    if 'page' in query_params:
        selected_page = query_params['page'][0].replace("+", " ")
        if selected_page in PAGES:
            st.session_state.current_page = selected_page
    else:
        st.session_state.current_page = "Home"


###############################
# PAGE: HOME
###############################
def home_page():
    st.markdown(
        """
        <div class="hero-section">
            <h1 class="hero-title">Cancer Support & Empowerment</h1>
            <p class="hero-subtitle">
                A comprehensive, high-end resource platform designed to assist newly diagnosed cancer patients and their families. 
                Explore the latest research, find top hospitals, manage finances, and discover emotional & social support – all in one place.
            </p>
            <img class="hero-image" src="https://www.cancer.org/content/dam/cancer-org/images/logos/cancerorg-logo.png" alt="Hero Image" />
        </div>
        """, unsafe_allow_html=True
    )

    st.write(
        """
        **Welcome to the Cancer Support Web Application**. This platform is designed to provide **comprehensive resources** and tools 
        to assist cancer patients and their families. Use the **top navigation bar** to explore:
        
        - **Locate Hospitals**: Find top-rated cancer hospitals near you.
        - **Accommodation Resources**: Discover lodging options during treatment.
        - **Latest Research**: Stay updated with the newest cancer research.
        - **Financial Support**: Learn about financial relief and legal rights.
        - **Clinical Trials**: Find relevant clinical trials for your condition.
        - **Emotional & Social Support**: Access mental health resources and support groups.
        - **Interactive Tools & Extras**: Utilize tools like checklists, donation hubs, etc.
        - **Symptom Checker**: Get quick, generalized suggestions (not a substitute for professional diagnosis).
        - **Medication Tracker**: Organize and schedule your medications.
        - **Personal Journal**: Keep a private log of your daily experiences and reflections.
        - **Appointment Scheduler**: Plan and keep track of medical appointments.
        - **AI Chatbot (Beta)**: Ask questions about general cancer information, treatments, and more.
        
        **Disclaimer**: This app is not a substitute for professional medical advice. Always consult qualified healthcare providers for 
        accurate guidance regarding your health.
        """
    )


###############################
# PAGE: LOCATE HOSPITALS
###############################
def locate_hospitals():
    st.title("Locate the Best Cancer Hospitals Nearby")
    st.markdown(
        """
        **Find top-rated cancer hospitals specializing in your area. Use the interactive map below to explore nearby facilities.**
        """
    )

    location = st.text_input("Enter your city or ZIP code:", "New York")

    if st.button("Find Hospitals"):
        with st.spinner("Searching for hospitals..."):
            # Define headers with User-Agent for Nominatim API
            headers = {
                "User-Agent": "CancerSupportApp/1.0 (support@example.com)"
            }

            # Geocode using Nominatim
            geocode_url = "https://nominatim.openstreetmap.org/search"
            geocode_params = {
                "q": location,
                "format": "json",
                "limit": 1
            }

            try:
                geocode_response = requests.get(geocode_url, headers=headers, params=geocode_params, timeout=10)
                geocode_response.raise_for_status()
                geocode_data = geocode_response.json()
            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error occurred during geocoding: {http_err}")
                return
            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again later.")
                return
            except requests.exceptions.RequestException as req_err:
                st.error(f"An error occurred during geocoding: {req_err}")
                return
            except ValueError:
                st.error("Received an invalid response from the geocoding service.")
                return

            if geocode_data:
                lat = geocode_data[0].get('lat')
                lon = geocode_data[0].get('lon')

                if not lat or not lon:
                    st.error("Could not retrieve latitude and longitude for the specified location.")
                    return

                try:
                    lat = float(lat)
                    lon = float(lon)
                except ValueError:
                    st.error("Invalid latitude or longitude values received.")
                    return

                # Overpass API to find hospitals
                overpass_url = "http://overpass-api.de/api/interpreter"
                overpass_query = f"""
                [out:json];
                (
                  node["amenity"="hospital"](around:50000,{lat},{lon});
                  way["amenity"="hospital"](around:50000,{lat},{lon});
                  relation["amenity"="hospital"](around:50000,{lat},{lon});
                );
                out center;
                """

                try:
                    overpass_response = requests.get(
                        overpass_url, 
                        params={'data': overpass_query}, 
                        headers=headers, 
                        timeout=10
                    )
                    overpass_response.raise_for_status()
                    overpass_data = overpass_response.json()
                except requests.exceptions.HTTPError as http_err:
                    st.error(f"HTTP error occurred while fetching hospitals: {http_err}")
                    return
                except requests.exceptions.Timeout:
                    st.error("The request to Overpass API timed out. Please try again later.")
                    return
                except requests.exceptions.RequestException as req_err:
                    st.error(f"An error occurred while fetching hospitals: {req_err}")
                    return
                except ValueError:
                    st.error("Received an invalid response from the Overpass API.")
                    return

                hospitals = []
                for element in overpass_data.get('elements', []):
                    tags = element.get('tags', {})
                    name = tags.get('name', 'Unnamed Hospital')
                    lat_h = element.get('lat') or (element.get('center', {}).get('lat') if element.get('center') else None)
                    lon_h = element.get('lon') or (element.get('center', {}).get('lon') if element.get('center') else None)
                    if lat_h and lon_h:
                        hospitals.append({
                            "Name": name,
                            "Latitude": lat_h,
                            "Longitude": lon_h
                        })

                if hospitals:
                    df_hospitals = pd.DataFrame(hospitals)

                    # Display on map
                    m = folium.Map(location=[lat, lon], zoom_start=12)
                    folium.Marker(
                        [lat, lon],
                        popup="Your Location",
                        icon=folium.Icon(color='red', icon='home')
                    ).add_to(m)

                    for idx, row in df_hospitals.iterrows():
                        folium.Marker(
                            [row['Latitude'], row['Longitude']],
                            popup=row['Name'],
                            icon=folium.Icon(color='blue', icon='plus-sign')
                        ).add_to(m)

                    folium_static(m, width=700, height=500)

                    st.subheader("List of Hospitals")
                    st.dataframe(df_hospitals)
                else:
                    st.error("No hospitals found within a 50km radius.")
            else:
                st.error("Location not found. Please try a different location.")


###############################
# PAGE: ACCOMMODATION RESOURCES
###############################
def accommodation_resources():
    st.title("Accommodation Resources")
    st.markdown("""
    **Find lodging solutions for patients and their families during treatment. Below are some recommended resources:**
    """)

    st.header("Ronald McDonald House Charities")
    st.markdown("""
    [Ronald McDonald House](https://www.rmhc.org/) provides a place for families to stay while their loved ones receive treatment.
    """)

    st.header("Local Support Housing Programs")
    st.markdown("""
    - **CancerCare Housing Assistance**: [CancerCare](https://www.cancercare.org/)
    - **Hospice Housing Programs**: [Hospice Foundation](https://hospicefoundation.org/)
    """)

    st.header("Low-Cost Hotels Near Treatment Centers")
    st.markdown("""
    - [Booking.com](https://www.booking.com/) - Filter by proximity to your treatment center.
    - [Airbnb](https://www.airbnb.com/) - Affordable lodging options.
    """)

    st.header("Booking Links")
    st.markdown("""
    - [Reserve a Ronald McDonald House](https://www.rmhc.org/find-a-house)
    - [CancerCare Housing Assistance](https://www.cancercare.org/support_resources/housing_assistance)
    """)


###############################
# PAGE: LATEST RESEARCH
###############################
def latest_research():
    st.title("Latest Research and AI-Driven Insights")
    st.markdown("""
    **Stay updated with the latest research, treatment advancements, and breakthroughs related to your specific cancer type.**
    """)

    cancer_type = st.text_input("Enter your cancer type (e.g., Breast Cancer):", "Breast Cancer")

    if st.button("Get Latest Research"):
        with st.spinner("Fetching latest research articles..."):
            # Fetch latest 10 articles from PubMed
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": cancer_type,
                "retmax": 10,
                "sort": "pub date",
                "retmode": "json"
            }
            response = requests.get(base_url, params=params).json()
            id_list = response['esearchresult']['idlist']

            if id_list:
                # Fetch article details
                fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "xml",
                    "rettype": "abstract"
                }
                fetch_response = requests.get(fetch_url, params=fetch_params).text

                # Parse XML response using xmltodict
                try:
                    data_dict = xmltodict.parse(fetch_response)
                    articles = data_dict.get('PubmedArticleSet', {}).get('PubmedArticle', [])

                    if isinstance(articles, dict):
                        articles = [articles]

                    st.markdown("### Latest Research Articles")
                    for article in articles:
                        title = article.get('MedlineCitation', {}).get('Article', {}).get('ArticleTitle', 'No Title')
                        pmid = article.get('MedlineCitation', {}).get('PMID', {}).get('#text', '')
                        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        st.markdown(f"#### [{title}]({link})")
                except Exception as e:
                    st.error("Error parsing research articles.")
            else:
                st.warning("No articles found for the specified cancer type.")

    st.markdown("---")
    st.header("Stay Informed")
    st.markdown("""
    **Subscribe to email notifications** to receive updates on newly published studies and breakthroughs.

    *Feature coming soon!*
    """)

    st.header("AI Chatbot Assistance")
    st.markdown("""
    **Have questions about the latest research or treatments?**

    Try our **AI Chatbot (Beta)** from the top navigation bar to ask general questions.
    """)


###############################
# PAGE: FINANCIAL SUPPORT
###############################
def financial_support():
    st.title("Financial Support and Legal Options")
    st.markdown("""
    **Access information on financial relief, legal rights, and assistance programs to help manage the financial burden of cancer treatment.**
    """)

    st.header("Corporate Angel Network")
    st.markdown("""
    **Details on Corporate Angel Network**:
    - **Free Flights for Patients**: Assistance with travel arrangements for treatment.
    - **How to Apply**: [Corporate Angel Network Application](https://apexlg.com/an-example-of-social-entrepreneurship-from-nbcs-shark-tank/)
    """)

    st.header("Tax-Free Retirement Withdrawals")
    st.markdown("""
    Stage IV patients can withdraw money from retirement accounts tax-free under specific conditions.

    **More Information**:
    - [Diana Award](https://diana-award.org.uk/)
    - [IRS Guidelines on Retirement Withdrawals](https://www.irs.gov/retirement-plans/retirement-plans-faqs-regarding-required-minimum-distributions)
    """)

    st.header("Insurance Navigation")
    st.markdown("""
    - **Understanding Coverage**: [Health Insurance Basics](https://www.healthcare.gov/glossary/)
    - **Assistance Programs for Uninsured Patients**: [CancerCare Assistance](https://www.cancercare.org/)
    """)

    st.header("Interactive Financial Calculator")
    st.markdown("Estimate potential savings, grants, or tax benefits based on your data.")

    with st.form("financial_calculator"):
        income = st.number_input("Enter your annual income ($):", min_value=0, value=50000, step=1000)
        retirement_withdraw = st.number_input("Enter amount to withdraw from retirement account ($):", min_value=0, value=10000, step=1000)
        submitted = st.form_submit_button("Calculate")

        if submitted:
            # Placeholder calculation: Assuming 0% tax for Stage IV withdrawals
            tax = 0  # Simplified logic for demonstration
            st.write(f"**Estimated Tax on Withdrawal:** ${tax}")
            st.success("Calculation completed. Please consult a financial advisor for accurate information.")


###############################
# PAGE: CLINICAL TRIALS
###############################
def clinical_trials():
    st.title("Clinical Trials Finder")
    st.markdown("""
    **Find relevant clinical trials based on your condition, location, and treatment phase. Participate in studies to access cutting-edge treatments.**
    """)

    # User Inputs
    cancer_type = st.text_input("Enter your cancer type (e.g., Lung Cancer):", "Lung Cancer")
    location = st.text_input("Enter your location or ZIP code:", "New York")
    phase = st.selectbox("Select Trial Phase:", ["All", "Phase 1", "Phase 2", "Phase 3", "Phase 4"])

    if st.button("Find Clinical Trials"):
        with st.spinner("Searching for clinical trials..."):
            # Refine the search query
            query = f"{cancer_type}[Condition] AND {location}[Location]"
            if phase != "All":
                query += f" AND {phase}[Phase]"

            base_url = "https://clinicaltrials.gov/api/query/study/search/brief"
            params = {
                "expr": query,
                "min_rnk": 1,
                "max_rnk": 20,
                "fmt": "xml"
            }

            headers = {
                "User-Agent": "CancerSupportApp/1.0 (support@example.com)"
            }

            try:
                response = requests.get(base_url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error occurred while fetching clinical trials: {http_err}")
                return
            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again later.")
                return
            except requests.exceptions.RequestException as req_err:
                st.error(f"An error occurred while fetching clinical trials: {req_err}")
                return

            # Parse the XML response
            try:
                data_dict = xmltodict.parse(response.content)
                studies = data_dict.get('clinical_studies', {}).get('clinical_study', [])

                if isinstance(studies, dict):
                    studies = [studies]

                if studies:
                    st.markdown("### Found Clinical Trials")
                    for study in studies:
                        title = study.get('official_title', 'No Title')
                        status = study.get('overall_status', 'Status Unknown')

                        location_info = study.get('location_countries', {}).get('location_country', [])
                        if isinstance(location_info, dict):
                            location_info = [location_info]
                        # Some trials won't have location_countries; handle gracefully
                        if location_info:
                            # Attempt to read location text
                            try:
                                locations = ", ".join([loc.get('location', 'Unknown') for loc in location_info])
                            except:
                                locations = "Unknown"
                        else:
                            locations = "Unknown"

                        phase_text = study.get('phase', 'N/A')
                        nct_id = study.get('id_info', {}).get('nct_id', '')
                        link = f"https://clinicaltrials.gov/ct2/show/{nct_id}" if nct_id else "#"

                        st.markdown(f"#### [{title}]({link})")
                        st.write(f"**Status:** {status}")
                        st.write(f"**Phase:** {phase_text}")
                        st.write(f"**Locations:** {locations}")
                        st.markdown("---")
                else:
                    st.warning("No clinical trials found for the given criteria.")
            except Exception as e:
                st.error("Error parsing clinical trials data.")

    st.markdown("---")
    st.header("Enrollment Guide")
    st.markdown("""
    **How to Enroll in a Trial**:
    1. **Consult Your Doctor**: Discuss eligibility and suitability.
    2. **Contact the Study Team**: Reach out via the provided links.
    3. **Understand the Commitment**: Review the study requirements and benefits.

    **Pros and Cons of Participation**:
    - **Pros**: Access to new treatments, close monitoring, contributing to research.
    - **Cons**: Possible side effects, time commitment, uncertain outcomes.
    """)


###############################
# PAGE: EMOTIONAL & SOCIAL SUPPORT
###############################
def emotional_social_support():
    st.title("Emotional and Social Support")
    st.markdown("""
    **Address mental health and community-building needs with the resources below.**
    """)

    st.header("Counseling Options")
    st.markdown("""
    - **American Cancer Society Counseling Services**: [Find a Counselor](https://www.cancer.org/treatment/support-programs-and-services/find-support.html)
    - **CancerCare Therapy Services**: [Access Therapy](https://www.cancercare.org/services/therapy)
    - **Psychology Today**: [Find a Therapist](https://www.psychologytoday.com/us/therapists/cancer)
    """)

    st.header("Support Groups")
    st.markdown("""
    - **Meetup**: [Cancer Support Groups](https://www.meetup.com/topics/cancer-support/)
    - **Cancer Support Community**: [Join a Group](https://www.cancersupportcommunity.org/join-a-group)
    - **Local Hospitals and Clinics**: Many offer in-person and virtual support groups.
    """)


###############################
# PAGE: INTERACTIVE TOOLS & EXTRAS
###############################
def interactive_tools_extras():
    st.title("Interactive Tools and Extras")
    st.markdown("""
    **Utilize the tools below to manage tasks and support your journey.**
    """)

    st.header("Checklist Generator")
    st.markdown("Create your personalized to-do list based on your needs.")

    with st.form("checklist_form"):
        financial_tasks = st.multiselect("Financial Tasks", [
            "Apply for insurance",
            "Meet with financial advisor",
            "Fill out tax forms",
            "Explore Corporate Angel Network",
            "Plan budget for treatments"
        ])
        medical_appointments = st.multiselect("Medical Appointments", [
            "Schedule doctor's visit",
            "Radiation therapy session",
            "Chemotherapy session",
            "Follow-up consultations",
            "Get second opinion"
        ])
        other_tasks = st.multiselect("Other Tasks", [
            "Call support group",
            "Arrange transportation",
            "Update personal documents",
            "Organize living space",
            "Plan meals"
        ])

        submitted = st.form_submit_button("Generate Checklist")

        if submitted:
            st.markdown("### Your Personalized Checklist")
            if financial_tasks:
                st.markdown("**Financial Tasks:**")
                for task in financial_tasks:
                    st.write(f"- [ ] {task}")
            if medical_appointments:
                st.markdown("**Medical Appointments:**")
                for task in medical_appointments:
                    st.write(f"- [ ] {task}")
            if other_tasks:
                st.markdown("**Other Tasks:**")
                for task in other_tasks:
                    st.write(f"- [ ] {task}")

    st.markdown("---")

    st.header("Donation Hub")
    st.markdown("""
    **Support patients in need by donating to reputable charities and crowdfunding platforms:**
    
    - **CancerCare**: [Donate](https://www.cancercare.org/donate)
    - **Ronald McDonald House Charities**: [Donate](https://www.rmhc.org/donate)
    - **GoFundMe**: [Create a Fundraiser](https://www.gofundme.com/)
    - **Crowdfunder**: [Start a Campaign](https://www.crowdfunder.com/)
    """)


###############################
# PAGE: SYMPTOM CHECKER
###############################
def symptom_checker():
    st.title("Symptom Checker (Beta)")
    st.markdown("""
    **Disclaimer**: This tool is for general informational purposes and not a substitute for professional medical advice. 
    Always consult a qualified healthcare provider for an accurate diagnosis.
    """)

    st.write("Select the symptoms you're experiencing from the list below:")

    possible_symptoms = [
        "Fatigue",
        "Unintentional Weight Loss",
        "Persistent Pain",
        "Fever",
        "Skin Changes",
        "Persistent Cough",
        "Unusual Bleeding",
        "Difficulty Swallowing",
        "Change in Bowel Habits"
    ]
    selected_symptoms = st.multiselect("Symptoms", possible_symptoms)

    if st.button("Check"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
            return

        # Very simple example of "analysis"
        # (Real systems would use advanced ML and medical knowledge.)
        suggestion = ""
        if "Unintentional Weight Loss" in selected_symptoms or "Persistent Pain" in selected_symptoms:
            suggestion += "- You have selected symptoms that may warrant a more urgent evaluation.\n"
        if "Fever" in selected_symptoms:
            suggestion += "- Persistent or recurring fever should be discussed with a doctor.\n"
        if "Difficulty Swallowing" in selected_symptoms:
            suggestion += "- Difficulty swallowing can be related to certain esophageal or throat issues.\n"

        if not suggestion:
            suggestion = "- Your selected symptoms are relatively common, but you should still consult a medical professional if they persist or worsen."

        st.markdown("### Preliminary Suggestions:")
        st.write(suggestion)


###############################
# PAGE: MEDICATION TRACKER
###############################
def medication_tracker():
    st.title("Medication Tracker")
    st.markdown("""
    Keep track of your medications, dosages, and schedules.
    """)

    if "medications" not in st.session_state:
        st.session_state.medications = []

    with st.form("med_form"):
        med_name = st.text_input("Medication Name")
        dosage = st.text_input("Dosage (e.g., 50mg)")
        frequency = st.text_input("Frequency (e.g., 2 times a day)")
        add_med = st.form_submit_button("Add Medication")

        if add_med:
            if med_name and dosage and frequency:
                st.session_state.medications.append({
                    "Medication": med_name,
                    "Dosage": dosage,
                    "Frequency": frequency
                })
                st.success("Medication added successfully!")
            else:
                st.error("Please fill in all fields.")

    if st.session_state.medications:
        st.markdown("### Your Current Medications")
        df_meds = pd.DataFrame(st.session_state.medications)
        st.dataframe(df_meds)

        # Option to remove medications
        remove_index = st.number_input("Enter the index of the medication to remove", min_value=0, 
                                       max_value=len(st.session_state.medications)-1, value=0)
        if st.button("Remove Selected Medication"):
            try:
                st.session_state.medications.pop(remove_index)
                st.success("Medication removed.")
            except IndexError:
                st.error("Invalid index. Cannot remove.")


###############################
# PAGE: PERSONAL JOURNAL
###############################
def personal_journal():
    st.title("Personal Journal")
    st.markdown("Write daily reflections, track emotional states, or record important thoughts.")

    if "journal_entries" not in st.session_state:
        st.session_state.journal_entries = []

    with st.form("journal_form"):
        entry_date = st.date_input("Entry Date", datetime.now())
        entry_text = st.text_area("Your Journal Entry")
        add_entry = st.form_submit_button("Save Entry")

        if add_entry:
            if entry_text.strip():
                st.session_state.journal_entries.append({
                    "date": entry_date,
                    "entry": entry_text
                })
                st.success("Journal entry saved!")
            else:
                st.error("Please write something in the journal entry.")

    if st.session_state.journal_entries:
        st.markdown("### Your Journal Entries")
        for idx, item in enumerate(sorted(st.session_state.journal_entries, key=lambda x: x["date"], reverse=True)):
            st.markdown(f"**{item['date']}**")
            st.write(item["entry"])
            st.markdown("---")


###############################
# PAGE: APPOINTMENT SCHEDULER
###############################
def appointment_scheduler():
    st.title("Appointment Scheduler")
    st.markdown("Organize your upcoming medical visits, therapy sessions, or check-ups.")

    if "appointments" not in st.session_state:
        st.session_state.appointments = []

    with st.form("appt_form"):
        appt_title = st.text_input("Appointment Title")
        appt_date = st.date_input("Date", datetime.now())
        appt_time = st.time_input("Time", datetime.now().time())
        submit_appt = st.form_submit_button("Add Appointment")

        if submit_appt:
            if appt_title.strip():
                st.session_state.appointments.append({
                    "title": appt_title,
                    "date": appt_date,
                    "time": appt_time
                })
                st.success("Appointment added successfully!")
            else:
                st.error("Please provide a title for the appointment.")

    if st.session_state.appointments:
        st.markdown("### Upcoming Appointments")
        sorted_appts = sorted(st.session_state.appointments, key=lambda x: (x["date"], x["time"]))
        for appt in sorted_appts:
            st.markdown(f"- **{appt['title']}** on **{appt['date']}** at **{appt['time']}**")


###############################
# PAGE: AI CHATBOT
###############################
def ai_chatbot():
    st.title("AI Chatbot (Beta)")
    st.markdown("""
    Ask general questions about cancer, treatments, resources, and more. 
    This chatbot uses a local **Question-Answering** model (DistilBERT) and a curated text context.  
    **Disclaimer**: Always consult a qualified professional for medical advice.
    """)

    # A sample context about cancer from reputable sources (shortened for demonstration)
    context = """
    Cancer is a group of diseases characterized by the uncontrolled growth and spread of abnormal cells. 
    There are over 100 different types of cancer. Treatment options vary and may include surgery, chemotherapy, radiation therapy, 
    targeted therapy, immunotherapy, or a combination of these. Early detection and accurate diagnosis are crucial for better outcomes. 
    Various support services exist to help patients cope with the physical, emotional, and financial challenges of cancer.
    """

    if nlp_qa is None:
        st.error("The QA model is not available. Please ensure 'transformers' is installed and try again.")
        return

    user_question = st.text_input("Enter your question about cancer:")
    if st.button("Get Answer"):
        if user_question.strip():
            with st.spinner("Thinking..."):
                try:
                    result = nlp_qa({
                        'question': user_question,
                        'context': context
                    })
                    st.write(f"**Answer**: {result['answer']}")
                except Exception as e:
                    st.error(f"An error occurred while processing your question: {e}")
        else:
            st.warning("Please enter a question.")


###############################
# MAIN RENDER FUNCTION
###############################
def render_page(page_name):
    if page_name == "Home":
        home_page()
    elif page_name == "Locate Hospitals":
        locate_hospitals()
    elif page_name == "Accommodation Resources":
        accommodation_resources()
    elif page_name == "Latest Research":
        latest_research()
    elif page_name == "Financial Support":
        financial_support()
    elif page_name == "Clinical Trials":
        clinical_trials()
    elif page_name == "Emotional & Social Support":
        emotional_social_support()
    elif page_name == "Interactive Tools & Extras":
        interactive_tools_extras()
    elif page_name == "Symptom Checker":
        symptom_checker()
    elif page_name == "Medication Tracker":
        medication_tracker()
    elif page_name == "Personal Journal":
        personal_journal()
    elif page_name == "Appointment Scheduler":
        appointment_scheduler()
    elif page_name == "AI Chatbot (Beta)":
        ai_chatbot()
    else:
        home_page()


def main():
    # Render the navbar
    navbar()
    # Render the current page
    render_page(st.session_state.current_page)

if __name__ == "__main__":
    main()
