import streamlit as st
import pandas as pd
import pickle as pkl

raw_data_set =  pkl.load(open('df2.pkl', 'rb'))
data_set = pd.DataFrame(raw_data_set)
st.header('Auto Clustering System')
st.write(data_set)

def main():
    st.title("Complaint Form")

    # Custom CSS styles
    st.markdown(
        """
        <style>
            /* Add your custom CSS styles here */
            .input-container {
                margin-bottom: 15px;
            }
            .input-label {
                font-weight: bold;
                font-size: 16px;
                color: #333333;
            }
            .input-field {
                width: 100%;
                padding: 8px;
                font-size: 14px;
                border-radius: 5px;
                border: 1px solid #CCCCCC;
            }
            .input-field:focus {
                outline: none;
                border-color: #007BFF;
            }
            .submit-button {
                background-color: #007BFF;
                color: white;
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            .submit-button:hover {
                background-color: #0056b3;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Name Input
    st.markdown('<div class="input-container"><label class="input-label">Name:</label><input class="input-field" type="text" placeholder="Enter your name"></div>', unsafe_allow_html=True)

    # Mobile Number Input
    st.markdown('<div class="input-container"><label class="input-label">Mobile Number:</label><input class="input-field" type="text" placeholder="Enter your mobile number"></div>', unsafe_allow_html=True)

    # Address Input
    st.markdown('<div class="input-container"><label class="input-label">Address:</label><textarea class="input-field" placeholder="Enter your address"></textarea></div>', unsafe_allow_html=True)

    # Complaint Input
    st.markdown('<div class="input-container"><label class="input-label">Complaint:</label><textarea class="input-field" placeholder="Enter your complaint"></textarea></div>', unsafe_allow_html=True)

    # Submit button
    st.button("Reset", type="primary")
    
        

if __name__ == "__main__":
    main()
