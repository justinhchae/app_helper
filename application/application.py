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
        st.title('A Data Helper App')

    def body(self):
        st.markdown("<h3 style='text-align: center; color: black;font-family:courier;'> upload a csv, get an optimized pandas pickle </h3>", unsafe_allow_html=True)
        DataLoader().read_data()

    def footer(self):
        st.markdown(
            '<i style="font-size:11px">alpha version 0.1</i>',
            unsafe_allow_html=True)

        st.markdown(
            '<i style="font-size:11px">&copy All Rights Reserved [@justinhchae](https://twitter.com/justinhchae?lang=en)</i>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:11px">The information provided by this app (the “Site”) is for general informational purposes only. All information on the Site is provided in good faith, however we make no representation or warranty of any kind, express or implied, regarding the accuracy, adequacy, validity, reliability, availability or completeness of any information on the Site.</p>',
            unsafe_allow_html=True
        )





