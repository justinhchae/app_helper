import streamlit as st

from do_data.data_loader import DataLoader

class Application():
    def __init__(self):
        st.set_page_config(page_title='Helper')

    def run_app(self):
        self.frame()

    def frame(self):
        self.title()
        self.body()

    def title(self):
        st.title('Data Helper')

    def body(self):
        st.markdown("<h3 style='text-align: center; color: black;font-family:courier;'> csv to cleaned pandas dataframe to pickle </h3>", unsafe_allow_html=True)
        st.markdown('**CSV Data Cleaner for Pandas**')
        DataLoader().read_data()




