import streamlit as st
st.title("Parkinson Disease Detection App")
st.sidebar.title("Devoleped By")
st.sidebar.write("- Fathima")
st.sidebar.write("- Sona")
st.sidebar.write("- Akash Lajish")
st.sidebar.write("- Adi S")

st.write('<b> About </b>',unsafe_allow_html=True)
st.write("""Parkinson's disease is a progressive nervous system disorder that affects movement. It develops gradually, often starting with a slight tremor in one hand. Over time, the disorder causes stiffness or slowing of movement. Other symptoms may include balance problems, speech changes, and muscle rigidity.

While the exact cause of Parkinson's disease is unknown, it is believed to involve a combination of genetic and environmental factors. Treatment typically focuses on managing symptoms through medication, lifestyle changes, and sometimes surgery.

Early diagnosis and intervention can help manage symptoms and improve quality of life for individuals with Parkinson's disease. Regular monitoring and appropriate medical care are essential for effective management of the condition.""")



st.header('Our Features')
st.markdown("""
            <b>Spiral Image Prediction:</b> Utilizes spiral images for predicting Parkinson's disease\n
            <b>Wave Pattern Image Prediction:</b> Utilizes wave pattern images for predicting Parkinson's disease.\n
            <b>Machine Learning Prediction:</b> Utilizes machine learning models trained on relevant features for predicting Parkinson's diseas""",
            unsafe_allow_html=True)
