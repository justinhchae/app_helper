import streamlit as st
import pandas as pd
# import docx2txt
# from PyPDF2 import PdfFileReader
# import pdfplumber
import numpy as np
import time
import gc
import base64
import io
import pickle
from textblob import TextBlob

from collections import defaultdict

from streamlit.report_thread import get_report_ctx
from streamlit import caching

class DataLoader():
    def __init__(self):
        self.upload_file = None
        self.df = None
        self.df_new = None
        self.index_col = None
        self.select_col = None
        self.usecols = None
        self.set_dtypes = None
        self.refresh = None
        self.standardize = None

        # https://stackoverflow.com/questions/26367812/appending-to-list-in-python-dictionary

    def read_data(self):
        label='start here'
        type=['csv']
        accept_multiple_files=False
        key='dataloader1'

        self.upload_file = st.file_uploader(label=label
                                       , type=type
                                       , accept_multiple_files=accept_multiple_files
                                       , key=key)

        if self.upload_file is not None:
            self.edit_data()
            #FIXME allow user to set index col 0
            # if self.index_col:
            #     index_col = 0
            # else:
            #     index_col = self.index_col

            with st.spinner('Reading and Parsing DataFrame Columns'):
                self.df = pd.read_csv(self.upload_file)
                self.df.columns = self.parse_cols(self.df)

            st.success('Ready! Parsed DataFrame Columns to lower case with underscores _.')

            if self.select_col:
                self.select_cols()

            st.markdown("<h3 style='text-align: center; color: black;'> File Preview </h3>", unsafe_allow_html=True)
            st.write(self.df.head())

            if self.df is not None:
                st.markdown("<h3 style='text-align: center; color: black;'> Optimize dtypes </h3>",
                            unsafe_allow_html=True)

                self.set_dtypes = st.checkbox('Optimize dtypes', value=False, key='dtypes_1')
                if self.set_dtypes:
                    self.optimizer()

            if self.set_dtypes and self.df_new is not None:
                self.standardize = st.checkbox('Standardize Data', value=False, key='standardize_1')

                if self.standardize:
                    st.markdown("<h3 style='text-align: center; color: black;'> Standardize Data </h3>",
                                unsafe_allow_html=True)
                    self.standardizer(self.df_new)

            if self.standardize:
                self.downloader()

    def edit_data(self):

        def options():
            col1, col2, col3 = st.beta_columns(3)
            with col1:
                if st.button('Refresh', key='clear_cache1'):
                    try:
                        if self.df is not None:
                            del self.df
                        # st.write('Refresh Session ID', get_report_ctx().session_id)
                    except:
                        pass

            with col2:
                # self.index_col = st.checkbox('Set Index Column = 0', value=False, key='1')
                pass
            with col3:
                self.select_col = st.checkbox('Select Specific Columns', value=False, key='2')

        def expander(key):
            options()

        my_expander = st.beta_expander("Edit Read CSV", expanded=True)
        with my_expander:
            clicked = expander("filter")

    def select_cols(self):
        if self.df is not None:
            try:
                df_columns = list(self.df.columns)
                label='Select Columns'
                self.usecols = st.multiselect(label=label, default=df_columns,options=df_columns)
                self.df = self.df[self.usecols]
            except:
                st.write('Column Select Error')

    def mem_usage(self, df):
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.
            source: https://gist.github.com/enamoria/fa9baa906f23d1636c002e7186516a7b
        """
        mem = df.memory_usage().sum() / 1024 ** 2
        return '{:.2f} MB'.format(mem)
    
    def reduce_precision(self, run_spellcheck=False):
        df = self.df.copy()
        cols_to_convert = []
        date_strings = ['_date', 'date_', 'date']

        for col in df.columns:
            col_type = df[col].dtype
            if 'string' not in col_type.name and col_type.name != 'category' and 'datetime' not in col_type.name:
                cols_to_convert.append(col)

        def _reduce_precision(x):
            col_type = x.dtype
            unique_data = list(x.unique())
            bools = [True, False, 'true', 'True', 'False', 'false']
            #TODO: account for only T or only F or 1/0 situations
            n_unique = float(len(unique_data))
            n_records = float(len(x))
            cat_ratio = n_unique / n_records

            try:
                unique_data.remove(np.nan)
            except:
                pass

            if 'int' in str(col_type):
                c_min = x.min()
                c_max = x.max()

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    x= x.astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    x = x.astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    x = x.astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    x = x.astype(np.int64)

                    # TODO: set precision to unsigned integers with nullable NA

            elif 'float' in str(col_type):
                c_min = x.min()
                c_max = x.max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        x = x.astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    x = x.astype(np.float32)
                else:
                    x = x.astype(np.float64)

            elif 'datetime' in col_type.name or any(i in str(x.name).lower() for i in date_strings):
                try:
                    x = pd.to_datetime(x)
                except:
                    pass

            elif any(i in bools for i in unique_data):
                x = x.astype('boolean')
                #TODO: set precision to bool if boolean not needed

            elif cat_ratio < .1 or n_unique < 20:
                try:
                    #TODO: work on standardizing text data within narrow edit distance
                    # x = x.fillna('Missing')
                    x = x.str.title()
                    # if run_spellcheck:
                    #     correct_spellings = [str(TextBlob(i).correct()).title() for i in list(x.unique())]
                    #     key = dict(zip(list(x.unique()), correct_spellings))
                    #     x = np.where(isinstance(x, str), x.map(key), x)
                except:
                    pass

                x = pd.Categorical(x)

            elif all(isinstance(i, str) for i in unique_data):
                x = x.astype('string')

            return x

        df[cols_to_convert] = df[cols_to_convert].apply(lambda x: _reduce_precision(x))

        return df
            
    def parse_cols(self, df):
        df.columns = map(str.lower, df.columns)
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('-', '_')
        return df.columns

    def optimizer(self):
        def options():
            with st.spinner('Optimizing DataFrame Memory'):
                self.df_new = self.reduce_precision()
                initial_memory = self.mem_usage(self.df)
                new_memory = self.mem_usage(self.df_new)

            st.success(str('Reduced memory from [' + initial_memory + '] to [' + new_memory + ']'))

        my_expander = st.beta_expander("Set dtypes", expanded=True)
        with my_expander:
            options()

        try:

            col1, col2 = st.beta_columns(2)
            with col1:
                st.write('Original dtypes')
                st.write(self.mem_usage(self.df))
                st.dataframe(self.df.dtypes)
            with col2:
                st.write('New dtypes')
                try:
                    st.write(self.mem_usage(self.df_new))
                    st.dataframe(self.df_new.dtypes)
                except:
                    st.write('Results Pending')

        except:
            st.write('Error Try Reloading the Page')

    def standardizer(self, df):

        if df is not None:
            types = df.dtypes
            test = list(zip(types, types.index))

            target_cols = []
            for type_id, col_name in test:
                if 'category' in str(type_id):
                    target_cols.append(col_name)

            if target_cols:

                select_cols = st.selectbox('Standardize Columns', target_cols)

                col1, col2, col3 = st.beta_columns(3)
                with col1:
                    st.write('Unique values in:', str(select_cols))
                    st.markdown(
                        "<h4 style='text-align: center; color: black;font-family:courier;'> all unique values in the column </h4>",
                        unsafe_allow_html=True)
                    st.write('')
                    categories = list(df[select_cols].dropna().astype('object').unique())

                    checkers = {}
                    for index, value in enumerate(categories):
                        checker = st.checkbox(value, value=True, key=str('std_' + str(index)))
                        checkers.update({value: checker})

                with col2:
                    st.write('Standardize:', str(select_cols))
                    st.markdown(
                        "<h4 style='text-align: center; color: black;font-family:courier;'> enter new values -> click Save Changes </h4>",
                        unsafe_allow_html=True)

                    display_value = [key for key, value in checkers.items() if value is True]
                    self.changers = {}

                    for index, value in enumerate(display_value):
                        changer = st.text_input(value, value=value)
                        self.changers.update({value: changer})

                with col3:
                    st.write('Actions:')
                    st.markdown(
                        "<h4 style='text-align: center; color: black;font-family:courier;'> save, clear, view changes </h4>",
                        unsafe_allow_html=True)

                    @st.cache(allow_output_mutation=True, persist=True)
                    def persist_list():
                        return []

                    @st.cache(allow_output_mutation=True, persist=True)
                    def persist_dict():
                        return defaultdict(list)

                    standardize_values_dict = persist_dict()

                    if self.changers:
                        if st.button('Save Changes'):
                            try:
                                #TODO: add session ID tag to df and changes to apply cashe per use
                                records = tuple((str(select_cols), self.changers))

                                curr_cols = [str(val[0]) for _, val in enumerate(standardize_values_dict[get_report_ctx().session_id])]

                                if curr_cols and any(x == str(select_cols) for x in curr_cols):
                                    st.write('Saved over previous values')
                                    new_list = [i for i in standardize_values_dict[get_report_ctx().session_id] if
                                                i[0] != str(select_cols)]
                                    standardize_values_dict[get_report_ctx().session_id] = new_list

                                standardize_values_dict[get_report_ctx().session_id].append(records)

                                st.write('Added new values for', str(select_cols))

                            except:
                                st.write('commit error, refresh and try again')

                    if st.button('Clear All Changes', key='clear_cache2'):
                        try:
                            standardize_values_dict.pop(get_report_ctx().session_id)
                            st.write('Cleared Changes in Current Session')
                        except:
                            pass

                    if st.button('View Current Changes', key='view_changes'):
                        if standardize_values_dict:
                            st.write(standardize_values_dict[get_report_ctx().session_id])
                        else:
                            st.write('No Changes Yet')

                self.standardize_values = standardize_values_dict

            else:
                st.write('No columns to standardize, skip to next section.')


    def make_categories(self, df, value_map):
        cols = []
        values = []

        for col, _ in value_map:
            cols.append(col)

        df[cols] = df[cols].apply(lambda x: x.astype('object'))

        for col, key in value_map:
            df[col] = df[col].map(key, na_action='ignore')

        df[cols] = df[cols].apply(lambda x: pd.Categorical(x))

        return df

    def downloader(self):
        def download():
            st.markdown("<h3 style='text-align: center; color: black;'> Download Data </h3>",
                        unsafe_allow_html=True)

            if st.button('Generate Dataframe'):
                try:
                    list_of_values = self.standardize_values[get_report_ctx().session_id]
                    self.df_new = self.make_categories(self.df_new, list_of_values)
                except:
                    st.write('Skipped Standardizer, writing dataframe without standardized values.')

                tmp_download_link = self.download_link(self.df_new, 'dataframe', 'Click here to download your data!')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.balloons()
                try:
                    self.standardize_values.pop(get_report_ctx().session_id)
                except:
                    pass

        downloader = st.beta_expander("Generate Pandas Pickle", expanded=True)
        with downloader:
            download()
            st.markdown("<h4 style='text-align: center; color: black;font-family:menlo;'> usage after download: </h4>",
                        unsafe_allow_html=True)
            st.markdown(
                "<h4 style='text-align: center; color: black;font-family:menlo;'> df = pd.read_pickle('df.pickle') </h4>",
                unsafe_allow_html=True)
            # print(self.df_new['item_name'].head())
            # TODO: generate hmac for data integrity check on download

    def download_link(self, df, download_filename, download_link_text):
        """
        Generates a link to download the given object_to_download.

        object_to_download (str, pd.DataFrame):  The object to be downloaded.
        download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
        download_link_text (str): Text to display for download link.

        Examples:
        download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
        download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')
        ref: https://discuss.streamlit.io/t/how-to-download-a-trained-model/2976

        """
        # df = df.sample(100000)
        dataframe = pickle.dumps(df, protocol=2)
        b64 = base64.b64encode(dataframe).decode()

        download_link = f'<a href="data:file/dataframe;base64,{b64}" download="df.pickle">Download DataFrame Pickle</a>'
        # st.markdown(download_link, unsafe_allow_html=True)
        return download_link












