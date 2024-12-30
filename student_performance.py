import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset with the correct delimiter
@st.cache_data
def load_data():
    return pd.read_csv("student-mat.csv", delimiter=';')  # Specify semicolon as delimiter

def add_jitter(data, jitter_strength=0.1):
    return data + np.random.uniform(-jitter_strength, jitter_strength, size=len(data))

# Load dataset
df = load_data()

# Streamlit title
st.title("Student Exam Performance Analysis")

# Create Tabs
tabs = [
    "Load Dataset",
    "Dataset Overview",
    "Dataset Information",
    "Summary Statistics",
    "Correlation Heatmap",
    "Boxplot of Numeric Features",
    "Interactive Scatter Plot"
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tabs)

# Tab 1: Dataset Overview
with tab1:
    st.subheader("Dataset Overview")
    st.write("This dataset contains information about student performance on final exams (G1, G2, G3).")
    st.write(f"**Number of rows:** {df.shape[0]}")
    st.write(f"**Number of columns:** {df.shape[1]}")
    st.write(f"**Columns in the dataset:** {', '.join(df.columns)}")
    st.write("You can explore the dataset and its various visualizations using the tabs above.")

# Tab 2: Display First Few Rows of the Dataset
with tab2:
    st.subheader("First Few Rows of the Dataset")
    st.write(df.head())

# Tab 3: Dataset Information
with tab3:
    st.subheader("Dataset Information")
    st.markdown("""
        ### Dataset Overview:
        This dataset contains **395 rows** and **33 columns**, with a mix of **numeric** and **categorical** data.

        **Columns Overview:**
        - **Numeric Columns (16)**: Includes columns like `age`, `Medu`, `studytime`, `G1`, `G2`, and `G3`.
        - **Categorical Columns (17)**: Includes columns like `school`, `sex`, `address`, `Mjob`, and `Fjob`.
    """)

    # Display Column Names and Data Types in a Table
    st.write("### Columns and Data Types")
    dataset_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(dataset_info)

    # Missing Values Section
    st.write("### Missing Values:")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])  # Display only columns with missing values

# Tab 4: Generate Summary Statistics
with tab4:
    st.subheader("Summary Statistics")
    
    # Numeric columns summary
    st.write("### Numeric Columns Summary:")
    st.write(df.describe())  # This will show stats for numeric columns (mean, std, min, 25%, 50%, 75%, max)
    
    # Categorical columns summary
    st.write("### Categorical Columns Summary:")
    categorical_columns = df.select_dtypes(include=['object']).columns
    st.write(df[categorical_columns].describe())  # This will show count, unique, top, freq for categorical columns

# Tab 5: Correlation Heatmap
with tab5:
    st.subheader("Correlation Heatmap")

    # Encode non-numeric columns using Label Encoding
    label_encoder = LabelEncoder()

    # Select columns that are not numeric (categorical)
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Apply Label Encoding to each categorical column
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Now create the correlation matrix, including both numeric and encoded categorical columns
    correlation_matrix = df.corr()

    # Create a figure and axis explicitly for the heatmap with a larger size
    fig, ax = plt.subplots(figsize=(24, 16))  # Increase the figure size here
    
    # Create the heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    
    # Display the plot in Streamlit
    st.pyplot(fig)

# Tab 6: Boxplot of Numeric and Non-Numeric Features
with tab6:
    st.subheader("Boxplot of Numeric Features")

    # Encode non-numeric columns using Label Encoding
    label_encoder = LabelEncoder()

    # Apply Label Encoding to each categorical column
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Select numeric columns for boxplot
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    all_columns = numeric_columns.tolist() + categorical_columns.tolist()  # Combine numeric and encoded categorical columns

    # 1. Boxplot for Numeric Features (already existing boxplot)
    fig1, ax1 = plt.subplots(figsize=(24, 16))
    sns.boxplot(data=df[numeric_columns], ax=ax1)
    st.pyplot(fig1)

    # 2. Boxplot for Non-Numeric (Encoded) Features (additional boxplot for categorical)
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df[categorical_columns], ax=ax2)
    st.pyplot(fig2)

    # 3. Combined Boxplot for All Features (Numeric and Non-Numeric)
    fig3 = go.Figure()
    for column in all_columns:
        fig3.add_trace(go.Box(
            y=df[column],
            name=column,
            boxmean='sd',
            hoverinfo='y+name'  # Show y-value and column name on hover
        ))

    st.plotly_chart(fig3)

    # Dropdown to select a specific boxplot (of either numeric or categorical)
    selected_column = st.selectbox("Select a column to display the boxplot:", all_columns)
    
    # Display the selected boxplot
    if selected_column:
        fig4 = go.Figure()
        fig4.add_trace(go.Box(
            y=df[selected_column],
            name=selected_column,
            boxmean='sd',
            hoverinfo='y+name'
        ))
        st.plotly_chart(fig4)
# Tab 7: Interactive Scatter Plot - User-defined Y-axis
with tab7:
    st.subheader("Interactive Scatter Plot - Choose Y-Axis")

    # Define the list of numeric columns for y-axis selection
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Default values for the X axis and Y axis
    default_x = 'Student Index'  # Default X-axis: sequence of student numbers
    default_y = 'G3'             # Default Y-axis: G3 (final exam score)

    # Create a dropdown for the Y axis selection
    y_axis = st.selectbox("Select Y-axis:", options=numeric_columns, index=list(numeric_columns).index(default_y))

    # Create a scatter plot with the student index on X-axis and user-selected feature on Y-axis
    df['Student Index'] = range(1, len(df) + 1)  # Add student index (1, 2, 3, ..., N)

    fig = px.scatter(df, x='Student Index', y=y_axis, 
                     title=f"Student Index vs {y_axis}", 
                     labels={'Student Index': 'Student Number', y_axis: y_axis},
                     hover_data=['age', 'failures', 'studytime', 'schoolsup'])  # Show additional info on hover
    
    # Show the plot
    st.plotly_chart(fig)