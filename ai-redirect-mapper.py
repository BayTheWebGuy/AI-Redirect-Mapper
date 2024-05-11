import base64
import chardet
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF, EditDistance, RapidFuzz
import plotly.graph_objects as go
import xlsxwriter

# BaileyDoesSEO | https://BaileyDoesSEO.com | 18th May 2024
# Inspired by LeeFootSEO | https://leefoot.co.uk | 10th December 2023


# Streamlit Interface Setup and Utilities ------------------------------------------------------------------------------

def setup_streamlit_interface():
    """
    Sets up the Streamlit interface for the Automatic Website Migration Tool.
    Configures the page layout, title, and adds creator information and instructions.
    """
    st.set_page_config(page_title="Automatic Website Migration Tool | LeeFoot.co.uk", layout="wide")
    st.title("Automatic Website Migration Tool")
    st.markdown("### Effortlessly migrate your website data")

    st.markdown(
        """
        <p style="font-style: italic;">
            Created by <a href="https://twitter.com/LeeFootSEO" target="_blank">LeeFootSEO</a> |
            <a href="https://leefoot.co.uk" target="_blank">More Apps & Scripts on my Website</a>
        </p>
        """,
        unsafe_allow_html=True
    )

    show_instructions_expander()


def create_file_uploader_widget(column, file_types):
    """
    Creates a file uploader widget in Streamlit.

    Args:
    column (str): A label indicating the type of file to be uploaded (e.g., "Live", "Staging").
    file_types (list): A list of acceptable file types for upload (e.g., ['csv', 'xlsx', 'xls']).

    Returns:
    streamlit.file_uploader: The file uploader widget.
    """
    file_type_label = "/".join(file_types).upper()  # Creating a string like "CSV/XLSX/XLS"
    return st.file_uploader(f"Upload {column} {file_type_label}", type=file_types)


def select_columns_for_data_matching(title, options, default_value, max_selections):
    """
    Creates a multi-select widget in Streamlit for selecting columns for data matching.

    Args:
    title (str): The title of the widget.
    options (list): A list of options (columns) to choose from.
    default_value (list): Default selected values.
    max_selections (int): Maximum number of selections allowed.

    Returns:
    list: A list of selected options.
    """
    st.write(title)
    return st.multiselect(title, options, default=default_value, max_selections=max_selections)


def show_warning_message(message):
    """
    Displays a warning message in the Streamlit interface.

    Args:
    message (str): The warning message to display.
    """
    st.warning(message)


def show_instructions_expander():
    """
    Creates an expander in the Streamlit interface to display instructions on how to use the tool.
    """
    instructions = (
        "- Crawl both the staging and live Websites using Screaming Frog SEO Spider.\n"
        "- Export the HTML as CSV Files.\n"
        "- Upload your 'Live' and 'Staging' CSV files using the file uploaders below.\n"
        "- By Default the app looks for columns named 'Address' 'H1-1' and 'Title 1' "
        "but they can be manually mapped if not found.\n"
        "- Select up to 3 columns that you want to match.\n"
        "- Click the 'Process Files' button to start the matching process.\n"
        "- Once processed, a download link for the output file will be provided.\n"
        "- Statistic such as median match score and a total mediam similarity score "
        "will be shown. Run the script with a different combination of columns to "
        "get the best score!"
    )
    with st.expander("How to Use This Tool"):
        st.write(instructions)


def create_page_footer_with_contact_info():
    """
    Adds a footer with contact information to the Streamlit page.
    """
    footer_html = (
        "<hr style='height:2px;border-width:0;color:gray;background-color:gray'>"
        "<p style='font-style: italic;'>Need an app? Need this run as a managed service? "
        "<a href='mailto:hello@leefoot.co.uk'>Hire Me!</a></p>"
    )
    st.markdown(footer_html, unsafe_allow_html=True)


def validate_uploaded_files(file1, file2):
    """
    Validates the uploaded files to ensure they are different.

    Args:
    file1 (UploadedFile): The first uploaded file.
    file2 (UploadedFile): The second uploaded file.

    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not file1 or not file2 or file1.getvalue() == file2.getvalue():
        show_warning_message(
            "Warning: The same file has been uploaded for both live and staging. Please upload different files.")
        return False
    return True


def create_file_uploader_widgets():
    """
    Creates file uploader widgets for live and staging files in the Streamlit interface.

    Returns:
    tuple: A tuple containing the live file uploader widget and the staging file uploader widget.
    """
    col1, col2 = st.columns(2)
    with col1:
        file_live = create_file_uploader_widget("Live", ['csv', 'xlsx', 'xls'])
    with col2:
        file_staging = create_file_uploader_widget("Staging", ['csv', 'xlsx', 'xls'])
    return file_live, file_staging


def handle_data_matching_and_processing(df_live, df_staging, address_column, selected_additional_columns,
                                        selected_model):
    """
    Handles the process of data matching and processing between live and staging dataframes.

    Args:
    df_live (pd.DataFrame): The dataframe for the live data.
    df_staging (pd.DataFrame): The dataframe for the staging data.
    address_column (str): The name of the address column to use for matching.
    selected_additional_columns (list): Additional columns selected for matching.
    selected_model (str): The name of the matching model to use.

    Returns:
    pd.DataFrame: The final processed dataframe after matching.
    """
    message_placeholder = st.empty()
    message_placeholder.info('Matching Columns, Please Wait!')

    rename_dataframe_column(df_live, address_column, 'Address')
    rename_dataframe_column(df_staging, address_column, 'Address')

    all_selected_columns = ['Address'] + selected_additional_columns
    progress_bar = st.progress(0)
    df_final = process_uploaded_files_and_match_data(df_live, df_staging, all_selected_columns, progress_bar,
                                                     message_placeholder,
                                                     selected_additional_columns, selected_model)
    return df_final


# File Reading and Data Preparation ------------------------------------------------------------------------------------

def read_excel_file(file, dtype):
    """
    Reads an Excel file into a Pandas DataFrame.

    Args:
    file (UploadedFile): The Excel file to read.
    dtype (str): Data type to use for the DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing the data from the Excel file.
    """
    return pd.read_excel_file(file, dtype=dtype)


def read_csv_file_with_detected_encoding(file, dtype):
    """
    Reads a CSV file with automatically detected encoding into a Pandas DataFrame.

    Args:
    file (UploadedFile): The CSV file to read.
    dtype (str): Data type to use for the DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    result = chardet.detect(file.getvalue())
    encoding = result['encoding']
    return pd.read_csv(file, dtype=dtype, encoding=encoding, on_bad_lines='skip')


def convert_dataframe_to_lowercase(df):
    """
    Converts all string columns in a DataFrame to lowercase.

    Args:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: DataFrame with all string columns in lowercase.
    """
    return df.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)


def rename_dataframe_column(df, old_name, new_name):
    """
    Renames a column in a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame containing the column to rename.
    old_name (str): The current name of the column.
    new_name (str): The new name for the column.

    Returns:
    None: The DataFrame is modified in place.
    """
    df.rename(columns={old_name: new_name}, inplace=True)


def process_and_validate_uploaded_files(file_live, file_staging):
    """
    Processes and validates the uploaded live and staging files.

    Args:
    file_live (UploadedFile): The live file uploaded by the user.
    file_staging (UploadedFile): The staging file uploaded by the user.

    Returns:
    tuple: A tuple containing the DataFrame for the live file and the DataFrame for the staging file.
    """
    if validate_uploaded_files(file_live, file_staging):
        # Determine file type and read accordingly
        if file_live.name.endswith('.csv'):
            df_live = read_csv_file_with_detected_encoding(file_live, "str")
        else:  # Excel file
            df_live = read_excel_file(file_live, "str")

        if file_staging.name.endswith('.csv'):
            df_staging = read_csv_file_with_detected_encoding(file_staging, "str")
        else:  # Excel file
            df_staging = read_excel_file(file_staging, "str")

        # Check if dataframes are empty
        if df_live.empty or df_staging.empty:
            show_warning_message("Warning: One or both of the uploaded files are empty.")
            return None, None
        else:
            return df_live, df_staging
    return None, None


# Data Matching and Analysis -------------------------------------------------------------------------------------------

def initialise_matching_model(selected_model="TF-IDF"):
    """
    Initializes the matching model based on the selected option.

    Args:
    selected_model (str, optional): The name of the model to use for matching. Defaults to "TF-IDF".

    Returns:
    PolyFuzz model: An instance of the selected PolyFuzz model.
    """
    if selected_model == "Edit Distance":
        from polyfuzz.models import EditDistance
        model = EditDistance()
    elif selected_model == "RapidFuzz":
        from polyfuzz.models import RapidFuzz
        model = RapidFuzz()
    else:  # Default to TF-IDF
        from polyfuzz.models import TFIDF
        model = TFIDF(min_similarity=0)
    return model


def setup_matching_model(selected_model):
    """
    Sets up the PolyFuzz matching model based on the selected model type.

    Args:
    selected_model (str): The name of the model to use for matching.

    Returns:
    PolyFuzz model: An instance of the selected PolyFuzz model.
    """
    if selected_model == "Edit Distance":
        model = PolyFuzz(EditDistance())
    elif selected_model == "RapidFuzz":
        model = PolyFuzz(RapidFuzz())
    else:  # Default to TF-IDF
        model = PolyFuzz(TFIDF())
    return model


def match_columns_and_compute_scores(model, df_live, df_staging, matching_columns):
    """
    Matches columns between two DataFrames (df_live and df_staging) and computes similarity scores.

    Args:
        model: The matching model to use for matching (e.g., PolyFuzz).
        df_live (pd.DataFrame): The DataFrame containing live data.
        df_staging (pd.DataFrame): The DataFrame containing staging data.
        matching_columns (list): List of column names to match between DataFrames.

    Returns:
        dict: A dictionary containing match scores for each column.
    """
    matches_scores = {}
    for col in matching_columns:
        # Check if the column exists in both dataframes
        if col in df_live.columns and col in df_staging.columns:
            # Ensure the data type is appropriate (i.e., Pandas Series)
            if isinstance(df_live[col], pd.Series) and isinstance(df_staging[col], pd.Series):
                live_list = df_live[col].fillna('').tolist()
                staging_list = df_staging[col].fillna('').tolist()

                # Here's the matching logic:
                model.match(live_list, staging_list)
                matches = model.get_matches()
                matches_scores[col] = matches

            else:
                st.warning(f"The column '{col}' in either the live or staging data is not a valid series.")
        else:
            st.warning(f"The column '{col}' does not exist in both the live and staging data.")

    return matches_scores


def identify_best_matching_url(row, matches_scores, matching_columns, df_staging):
    """
    Identifies the best matching URL for a given row in the DataFrame.

    Args:
    row (pd.Series): A row from the DataFrame.
    matches_scores (dict): Dictionary containing match scores for columns.
    matching_columns (list): List of column names used for matching.
    df_staging (pd.DataFrame): The DataFrame containing staging data.

    Returns:
    tuple: A tuple containing best match information and similarity scores.
    """
    best_match_info = {'Best Match on': None, 'Highest Matching URL': None,
                       'Highest Similarity Score': 0, 'Best Match Content': None}
    similarities = []

    for col in matching_columns:
        matches = matches_scores.get(col, pd.DataFrame())
        if not matches.empty:
            match_row = matches.loc[matches['From'] == row[col]]
            if not match_row.empty:
                similarity_score = match_row.iloc[0]['Similarity']
                similarities.append(similarity_score)
                if similarity_score > best_match_info['Highest Similarity Score']:
                    best_match_info.update({
                        'Best Match on': col,
                        'Highest Matching URL':
                            df_staging.loc[df_staging[col] == match_row.iloc[0]['To'], 'Address'].values[0],
                        'Highest Similarity Score': similarity_score,
                        'Best Match Content': match_row.iloc[0]['To']
                    })

    best_match_info['Median Match Score'] = np.median(similarities) if similarities else None
    return best_match_info, similarities


def add_additional_info_to_match_results(best_match_info, df_staging, selected_additional_columns):
    """
    Adds additional information to the best match results.

    Args:
    best_match_info (dict): Dictionary containing information about the best match.
    df_staging (pd.DataFrame): The DataFrame containing staging data.
    selected_additional_columns (list): List of additional columns selected for matching.

    Returns:
    dict: Updated best match information with additional details.
    """
    for additional_col in selected_additional_columns:
        if additional_col in df_staging.columns:
            staging_value = df_staging.loc[
                df_staging['Address'] == best_match_info['Highest Matching URL'], additional_col].values
            best_match_info[f'Staging {additional_col}'] = staging_value[0] if staging_value.size > 0 else None
    return best_match_info


def identify_best_matching_url_and_median(df_live, df_staging, matches_scores, matching_columns,
                                          selected_additional_columns):
    """
    Identifies the best matching URLs and computes median match scores for the entire DataFrame.

    Args:
    df_live (pd.DataFrame): The DataFrame containing live data.
    df_staging (pd.DataFrame): The DataFrame containing staging data.
    matches_scores (dict): Dictionary containing match scores for columns.
    matching_columns (list): List of column names used for matching.
    selected_additional_columns (list): List of additional columns selected for matching.

    Returns:
    pd.DataFrame: DataFrame with best match URLs and median scores.
    """

    def process_row(row):
        best_match_info, similarities = identify_best_matching_url(row, matches_scores, matching_columns, df_staging)
        best_match_info = add_additional_info_to_match_results(best_match_info, df_staging, selected_additional_columns)
        # Convert scores to percentage format with '%' sign
        best_match_info['All Column Match Scores'] = [
            (col, f"{round(score * 100)}%" if not pd.isna(score) else "NaN%")
            for col, score in zip(matching_columns, similarities)
        ]
        return pd.Series(best_match_info)

    return df_live.apply(process_row, axis=1)


def finalise_match_results_processing(df_live, df_staging, matches_scores, matching_columns,
                                      selected_additional_columns):
    """
    Finalizes the match result processing by combining live and matched data.

    Args:
    df_live (pd.DataFrame): The DataFrame containing live data.
    df_staging (pd.DataFrame): The DataFrame containing staging data.
    matches_scores (dict): Dictionary containing match scores for columns.
    matching_columns (list): List of column names used for matching.
    selected_additional_columns (list): List of additional columns selected for matching.

    Returns:
    pd.DataFrame: The final DataFrame after processing match results.
    """
    match_results = identify_best_matching_url_and_median(df_live, df_staging, matches_scores, matching_columns,
                                                          selected_additional_columns)
    df_final = prepare_concatenated_dataframe_for_display(df_live, match_results, matching_columns)
    return df_final


def process_uploaded_files_and_match_data(df_live, df_staging, matching_columns, progress_bar, message_placeholder,
                                          selected_additional_columns,
                                          selected_model):
    """
    Processes the uploaded files and performs data matching using the specified model.

    Args:
    df_live (pd.DataFrame): The DataFrame containing live data.
    df_staging (pd.DataFrame): The DataFrame containing staging data.
    matching_columns (list): List of column names to match between DataFrames.
    progress_bar (streamlit.progress_bar): Streamlit progress bar object.
    message_placeholder (streamlit.empty): Streamlit placeholder for messages.
    selected_additional_columns (list): Additional columns selected for matching.
    selected_model (str): The name of the matching model to use.

    Returns:
    pd.DataFrame: The final DataFrame after processing and matching data.
    """
    df_live = convert_dataframe_to_lowercase(df_live)
    df_staging = convert_dataframe_to_lowercase(df_staging)

    model = setup_matching_model(selected_model)
    matches_scores = process_column_matches_and_scores(model, df_live, df_staging, matching_columns)

    for index, _ in enumerate(matching_columns):
        progress = (index + 1) / len(matching_columns)
        progress_bar.progress(progress)

    message_placeholder.info('Finalising the processing. Please Wait!')
    df_final = finalise_match_results_processing(df_live, df_staging, matches_scores, matching_columns,
                                                 selected_additional_columns)

    display_final_results_and_download_link(df_final, 'migration_mapping_data.xlsx')
    message_placeholder.success('Complete!')

    return df_final


def scale_median_match_scores_to_percentage(df_final):
    """
    Scales the median match scores in the DataFrame to percentage values.

    Args:
    df_final (pd.DataFrame): The final DataFrame containing match results.

    Returns:
    pd.DataFrame: The DataFrame with scaled median match scores.
    """
    df_final['Median Match Score Scaled'] = df_final['Median Match Score'] * 100
    return df_final


def group_median_scores_into_brackets(df_final):
    """
    Groups the median match scores in the DataFrame into predefined score brackets.

    Args:
    df_final (pd.DataFrame): The final DataFrame containing match results.

    Returns:
    pd.DataFrame: The DataFrame with median scores grouped into brackets.
    """
    bins = range(0, 110, 10)
    labels = [f'{i}-{i + 10}' for i in range(0, 100, 10)]
    df_final['Score Bracket'] = pd.cut(df_final['Median
