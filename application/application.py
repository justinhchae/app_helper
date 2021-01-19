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
        self.footer()

    def title(self):
        st.title('A Data Helper App (alpha)')

    def body(self):
        st.markdown("<h3 style='text-align: center; color: black;font-family:courier;'> upload a csv, get an optimized pandas pickle </h3>", unsafe_allow_html=True)
        DataLoader().read_data()

    def footer(self):
        st.markdown(
            "<i>&copy All Rights Reserved [@justinhchae](https://twitter.com/justinhchae?lang=en)</i>",
            unsafe_allow_html=True)





