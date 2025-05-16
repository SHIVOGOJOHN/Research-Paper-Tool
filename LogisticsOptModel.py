import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import joblib

st.set_page_config(page_title="Logistics Optimization", layout="wide")

# Load pre-trained models
emission_model = joblib.load('trained_emission_model.joblib')

scaler = joblib.load('emissions_scaler.joblib')

# Preprocessing functions
def preprocess_emissions_data(df, training= False):


    # Create new features
    df['fuel_efficiency'] = df['distance_km'] / df['fuel_consumed_liters']
    
    # Create emission_intensity only if this is training data
    if training and 'estimated_emissions_kg' in df.columns:
        df['emission_intensity'] = df['estimated_emissions_kg'] / df['distance_km']
    else:
        # For prediction, set emission_intensity to 0 or another default value
        df['emission_intensity'] = 0
    
    # Encoding categorical variables
    traffic_level_map = {'low': 1, 'medium': 2, 'high': 3}
    df['traffic_level'] = df['traffic_level'].map(traffic_level_map)

    # Select and scale features
    features = ['distance_km', 'average_speed_kmph', 'fuel_consumed_liters', 'elevation_change_m', 'cargo_weight_tons', 'traffic_level',
                'vehicle_type_air_cargo', 'vehicle_type_cargo_ship', 'vehicle_type_diesel_truck', 'vehicle_type_electric_truck', 'vehicle_type_freight_train',
                'fuel_type_aviation_fuel', 'fuel_type_diesel', 'fuel_type_electric', 'emission_intensity', 'fuel_efficiency']
    
    # Ensure all expected columns are present
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0  # Add missing columns with 0 values
    
    # Select only the necessary features
    df = df[features]
    
    return scaler.transform(df)


# Similar preprocessing functions for other categories...

# Visualization Page
def visualization_page():
    st.title("Analysis Results Visualisations")

    # Display saved figures from session state
    if 'emission_fig' in st.session_state:
        st.pyplot(st.session_state.emission_fig)
    
    if st.button("Return to Home"):
        st.session_state.page = 'home'
        st.rerun()


def scroll_to_top():
    st.markdown("""
        <div style="position: fixed; bottom: 10px; right: 10px;">
            <a href="#logistics-optimization">
                <button style="
                    background: #000000;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 50%;
                    cursor: pointer;
                ">▲</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# --- UI Enhancements ---
def add_background():
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("https://raw.githubusercontent.com/SHIVOGOJOHN/Research-Paper-Tool/main/static/images/background4.jpg");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                -webkit-background-size: cover;
                -moz-background-size: cover;
                -o-background-size: cover;
            }}
            .main {{
                background: rgba(255, 255, 255, 0.95);
                padding: 2rem;
                border-radius: 10px;
                backdrop-filter: blur(5px);
            }}
        </style>
    """, unsafe_allow_html=True)

# main application
def main():
    scroll_to_top()
    add_background()

    #Css styling for buttons
    st.markdown(
    """
    <style>
    div.stButton > button {
        background-color:#0668DE;  /* Button background color */
        color: white;  /* Button text color */
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #0668DE;  /* No change on hover */
        border-color: none;      /* No change on hover */
        color:white;             /* No change on hover */
    }
    div.stButton > button:active {
        background-color: #de7006;  /* No change on click */
        border-color: none;      /* No change on click */
        color: white;             /* No change on click */
    }
    </style>
    """,
    unsafe_allow_html=True
    )

#####
    st.markdown(
    """
    <style>
    div.stDownloadButton > button {
        background-color:#0668DE;  /* Button background color */
        color: #000000;  /* Button text color */
        cursor: pointer;
    }
    div.stDownloadButton > button:hover {
        background-color: #0668DE;  /* No change on hover */
        border-color: none;      /* No change on hover */
        color: #000000;             /* No change on hover */
    }
    div.stDownloadButton > button:active {
        background-color: #0668DE;  /* No change on click */
        border-color: none;      /* No change on click */
        color: #000000;             /* No change on click */
    }
    </style>
    """,
    unsafe_allow_html=True
    )

### 
    st.markdown(
    """
    <style>
    /* Style the file uploader */
    .stFileUploader > div > div:first-child {
        border: 2px dashed #0668DE;
        border-radius: 5px;
        background-color: #f9f9f9;
        padding: 20px;
    }

    /* Style the number input widgets */
    div.stNumberInput > label {
        color: #white;
        font-weight: bold;
    }
    div.stNumberInput > div > div > input {
        border: 1px solid #0668DE;
        border-radius: 3px;
        padding: 5px;
    }

    /* Style the selectbox widgets */
    div.stSelectbox > label {
        color: white;
        font-weight: bold;
    }
    div.stSelectbox > div > div > div {
        border: 1px solid #0668DE;
        border-radius: 3px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
"""
<style>
/* Styling for the text input field */
.stNumberInput input {
    background-color: ##0668DE; /* Light blue background color */
    border: 2px solid ##0668DE; /* Green border color */
    border-radius: 5px; /* Rounded 
    padding: 10px; /* Space inside the input */
    font-size: 16px; /* Text size */
    color: white; /* Text color */
}

/* Placeholder text style */
.stNumberInput input::placeholder {
    color: #white; /* Gray placeholder text color */
    font-style: italic; /* Italic style for placeholder */
}

/* Focus state for the text input field */
.stNumberInput input:focus {
    border: 2px solid #0668DE; /* Change border color to blue on focus */
    outline: none; /* Remove default outline */
    background-color: #0668DE; /* Slightly darker background on focus */
}
</style>
""", unsafe_allow_html=True)
 
 
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if st.session_state.page == 'home':
        st.title("Logistics Optimization")
        
        # File upload section
        st.markdown(
            """
            <style>
            .stDropzone {
            border: 2px dashed #0668DE;
            border-radius: 5px;
            background-color: #f9f9f9;
            padding: 20px;
            }
            .stDropzone:hover {
            background-color: #f0f0f0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            # Check if the uploaded file has the required columns
            required_columns = ['distance_km', 'average_speed_kmph', 'fuel_consumed_liters', 'elevation_change_m', 'cargo_weight_tons', 'traffic_level',
                'vehicle_type_air_cargo', 'vehicle_type_cargo_ship', 'vehicle_type_diesel_truck', 'vehicle_type_electric_truck', 'vehicle_type_freight_train',
                'fuel_type_aviation_fuel', 'fuel_type_diesel', 'fuel_type_electric', 'emission_intensity', 'fuel_efficiency']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Uploaded file is missing the following columns: {', '.join(missing_columns)}")
            else:
                # Process the uploaded data
                df = preprocess_emissions_data(df, training=False)
                # Save the original data for later use
                st.session_state.original_data = pd.read_csv(uploaded_file)
                # Process the original data
                original_df = preprocess_emissions_data(st.session_state.original_data, training=False)
                # Make predictions using the original data
                emission_predictions = emission_model.predict(original_df)
                # Store the predictions in session state
                st.session_state.emission_predictions = emission_predictions
                # Display the predictions
                st.write("Predicted Emissions for the uploaded data:")
                st.write(emission_predictions)


            #st.session_state.data = df
            st.success("File uploaded successfully!")
        
        col1, col2 = st.columns(2)

        with col1:
            with open("sample_logistics_data.csv", "rb") as file:

                btn = st.download_button(
                    label="Download CSV Template",
                data=file,
                file_name="logistics_data_template.csv",
                mime="text/csv",
                help="Download a CSV template for data collection.",
                use_container_width=True,
                type="primary",
                key="csv_template_button"
                )
        
        with col2:
            with open("dataset_description.pdf", "rb") as file:
                btn = st.download_button(
                    label="Download Feature Description (PDF)",
                    data=file,
                    file_name="feature_description.pdf",
                    mime="application/octet-stream",
                    help="Download a PDF containing descriptions of the dataset features.",
                    use_container_width=True,
                    type="primary",
                    key="pdf_description_button"
                )
        

        # Category selection
        category = st.radio("Select Analysis Category:", 
                        ["Predict Carbon Emissions", 
                            "Optimize Travel Routes",
                            "Demand Forecasting"])
        
        # Category-specific input handling
        if category == "Predict Carbon Emissions":
            st.header("Carbon Emissions Prediction")
            
            if uploaded_file:
                #Automated processing for fie data
                processed_data = preprocess_emissions_data(st.session_state.data)
                prediction = emission_model.predict(processed_data)
                st.session_state.results = prediction
            else:
                #Manual Input
                distance = st.number_input("Distance(km)", min_value = 0.0, key="f--{distance}")
                fuel_consumed = st.number_input("Fuel Consumed (liters)", min_value=0.0, key="f--{fuel}")
                avg_speed = st.number_input("Average Speed (km/h)", min_value=0.0, key="f--{speed}")
                traffic_level = st.selectbox("Traffic Level", ["low", "medium", "high"])
                cargo_weight_tons = st.number_input("Cargo Weight(tons)", min_value = 0.0, key="f--{weight}")
                elevation_change_m = st.number_input("Elevation Change" , key="f--{elevation}")
                fuel_type = st.selectbox('Fuel Type', ["aviation_fuel", "diesel", "electric"])
                vehicle_type = st.selectbox('Vehicle Type', ["air_cargo", "ship_cargo", "diesel_truck", "electric_truck", "freight_train" ])
                emission_intensity = 0

                if fuel_consumed > 0:
                    fuel_efficiency = distance / fuel_consumed
                    st.write("Fuel Efficiency is: "
                            f"{fuel_efficiency:.2f} km/liters")
                else:
                    fuel_efficiency = 0
                    st.error("Fuel consumed must be greater than 0")
                

                if st.button("Calculate Emissions"):
                    input_data = pd.DataFrame({
                        'distance_km': [distance],
                        'cargo_weight_tons': [cargo_weight_tons],
                        'elevation_change_m': [elevation_change_m],
                        'average_speed_kmph': [avg_speed],
                        'fuel_consumed_liters': [fuel_consumed],
                        'fuel_type': [fuel_type],
                        'vehicle_type': [vehicle_type],
                        'traffic_level': [traffic_level],
                        'fuel_efficiency': [fuel_efficiency],
                        #'emission_intensity' : [emission_intensity]
                        })
                    
                    
                    # Apply one-hot encoding
                    encoded_data = pd.get_dummies(input_data, columns=['vehicle_type', 'fuel_type'], prefix=['vehicle_type', 'fuel_type'])
                    expected_vehicle_types = ['vehicle_type_air_cargo', 'vehicle_type_cargo_ship', 
                            'vehicle_type_diesel_truck', 'vehicle_type_electric_truck', 
                            'vehicle_type_freight_train']
                    expected_fuel_types = ['fuel_type_aviation_fuel', 'fuel_type_diesel', 'fuel_type_electric']
                    
                    
                    # Add missing columns with 0s
                    for col in expected_vehicle_types + expected_fuel_types:
                        if col not in encoded_data.columns:
                            encoded_data[col] = 0
                    
                    # Add emission_intensity (0 for prediction)
                    encoded_data['emission_intensity'] = 0

                    # Debug output
                    st.write("Debug - Processed features:")
                    st.write(encoded_data)

                    input_data = preprocess_emissions_data(encoded_data, training=False)
                    prediction = emission_model.predict(input_data)
                    st.session_state.results = prediction
            if 'results' in st.session_state:
                try:
                    if isinstance(st.session_state.results, (list, np.ndarray)):
                        st.write(f"Estimated Emissions: {st.session_state.results[0]:.2f} kg CO2")
                    else:
                        st.error("Invalid emission results format.")
                except (IndexError, TypeError) as e:
                    st.error(f"Error processing emission results: {e}")
            else:
                st.info("Emission results are not available. Please calculate emissions first.")
            
            if st.button("Visualize Results", key="viz_results"):
                if 'results' in st.session_state and st.session_state.results is not None:
                    # Create multiple tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs([
                        "Emissions Overview",
                        "Efficiency Analysis",
                        "Comparative Analysis"
                    ])
                
                    with tab1:
                        # Basic emissions bar plot
                        fig1, ax1 = plt.subplots(figsize=(15, 6))
                        sns.barplot(
                            x=['Predicted Emissions'],
                            y=[st.session_state.results[0]],
                            ax=ax1,
                            color='skyblue'  # Added color
                        )
                        ax1.set_ylabel('Emissions (kg CO2)', fontsize=12)  # Increased font size
                        ax1.set_title('Predicted Emissions Overview', fontsize=14)  # Increased font size
                        ax1.tick_params(axis='y', labelsize=10)  # Adjusted tick label size
                        st.pyplot(fig1)
                    
                    with tab2:
                        
                        # Efficiency Analysis
                        if 'distance' in locals() and 'fuel_consumed' in locals():
                            efficiency_data = {
                                'Distance (km)': distance,
                                'Fuel (Liters)': fuel_consumed
                            }
                            fig2, ax2 = plt.subplots(figsize=(15, 6))
                            sns.barplot(
                                x=list(efficiency_data.keys()),
                                y=list(efficiency_data.values()),
                                ax=ax2,
                                palette='viridis'  # Added color palette
                            )
                            ax2.set_ylabel('Value', fontsize=12)  # Increased font size
                            ax2.set_title('Distance and Fuel Consumption', fontsize=14)  # Increased font size
                            ax2.tick_params(axis='y', labelsize=10)  # Adjusted tick label size
                            st.pyplot(fig2)
                            
                        # Emissions intensity gauge chart
                            emissions_per_km = st.session_state.results[0] / distance if distance > 0 else 0
                            st.metric(
                                "Emissions Intensity",
                                f"{emissions_per_km:.2f} kg CO2/km"
                            )
                        else:
                            st.error("Distance and Fuel Consumed are required for Efficiency Analysis.")  
                    
                    with tab3:

                        # Percentage difference from industry average
                        percent_diff = ((st.session_state.results[0] - 500) / 500) * 100
                        st.metric("Comparison to Industry Average",
                                    f"{st.session_state.results[0]:.1f} kg CO2",
                                    f"{percent_diff:.1f}%")
                        
                        # Comparative Analysis
                        industry_averages = {
                            'Your Shipment': st.session_state.results[0],
                            'Industry Average': 500,
                            'Best Practice': 300
                        }
                        fig3, ax3 = plt.subplots(figsize=(15, 6))
                        sns.barplot(x=list(industry_averages.keys()),
                                    y=list(industry_averages.values()),
                                    ax=ax3,
                                    palette='muted')  # Added color palette
                        ax3.set_ylabel('Emissions (kg CO2)', fontsize=12)  # Increased font size
                        ax3.set_title('Comparative Emissions Analysis', fontsize=14)  # Increased font size
                        ax3.tick_params(axis='y', labelsize=10)  # Adjusted tick label size
                        st.pyplot(fig3)
                        
                        
                    
                    # Store figures in session state (optional, if needed later)
                    st.session_state.emission_fig = fig1
                    # st.session_state.page = 'visualization' #No need to change page
                    # st.rerun() #No need to rerun
                else:
                    st.error("No results available to visualize. Please calculate emissions first.")
    
    # elif st.session_state.page == 'visualization':
    #     visualization_page() #No need for this page

if __name__ == "__main__":
    main()




