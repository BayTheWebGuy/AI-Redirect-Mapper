import base64
import chardet
import io
import numpy as np
import streamlit as st
import pandas as pd
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF, EditDistance, RapidFuzz
import xlsxwriter
from sentence_transformers import SentenceTransformer
import faiss

# BaileyDoesSEO | https://BaileyDoesSEO.com | 18th May 2024
# Inspired by LeeFootSEO | https://leefoot.co.uk | 10th December 2023

# Streamlit Interface Setup and Utilities ------------------------------------------------------------------------------

def setup_streamlit_interface():
    st.set_page_config(page_title="Automatic Website Migration Tool", layout="wide")
    st.title("Automatic Website Migration Tool")
    st.markdown("### Effortlessly migrate your website data")

    st.markdown(
        """
        <p style="font-style: italic;">
            Embedding Model added by BaileyDoesSEO | Inspired by Lee Foot's original <a href="https://leefoot.co.uk/portfolio/automatic-website-migration-streamlit-app/">TF-IDF tool</a>.
        </p>
        """,
        unsafe_allow_html=True
    )

    show_instructions_expander()


def create_file_uploader_widget(column, file_types):
    file_type_label = "/".join(file_types).upper()  # Creating a string like "CSV/XLSX/XLS"
    return st.file_uploader(f"Upload {column} {file_type_label}", type=file_types)


def show_warning_message(message):
    st.warning(message)


def show_instructions_expander():
    instructions = (
        "- Crawl both the staging and live Websites using Screaming Frog SEO Spider.\n"
        "- Export the HTML as CSV Files.\n"
        "- Upload your 'Live' and 'Staging' CSV files using the file uploaders below.\n"
        "- Select up to 4 columns that you want to match. The **first** column you select will act as your primary identifier.\n"
        "- Click the 'Process Files' button to start the matching process.\n"
        "- Once processed, a download link for the output file will be provided.\n"
        "- Statistics such as median match score will be shown. Run the script with a different combination of columns to get the best score!"
    )
    with st.expander("How to Use This Tool"):
        st.write(instructions)


def create_page_footer_with_contact_info():
    pass  # Footer code removed as per the comment in the original script


def validate_uploaded_files(file1, file2):
    if not file1 or not file2 or file1.getvalue() == file2.getvalue():
        show_warning_message(
            "Warning: The same file has been uploaded for both live and staging. Please upload different files.")
        return False
    return True


def create_file_uploader_widgets():
    col1, col2 = st.columns(2)
    with col1:
        file_live = create_file_uploader_widget("Live", ['csv', 'xlsx', 'xls'])
    with col2:
        file_staging = create_file_uploader_widget("Staging", ['csv', 'xlsx', 'xls'])
    return file_live, file_staging


def handle_data_matching_and_processing(df_live, df_staging, matching_columns, selected_model):
    message_placeholder = st.empty()
    message_placeholder.info('Matching Columns, Please Wait!')

    progress_bar = st.progress(0)
    df_final = process_uploaded_files_and_match_data(
        df_live, df_staging, matching_columns, progress_bar,
        message_placeholder, selected_model
    )
    return df_final


# File Reading and Data Preparation ------------------------------------------------------------------------------------

def read_excel_file(file, dtype):
    return pd.read_excel(file, dtype=dtype)


def read_csv_file_with_detected_encoding(file, dtype):
    result = chardet.detect(file.getvalue())
    encoding = result['encoding']
    return pd.read_csv(file, dtype=dtype, encoding=encoding, on_bad_lines='skip')


def convert_dataframe_to_lowercase(df):
    return df.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)


def process_and_validate_uploaded_files(file_live, file_staging):
    if validate_uploaded_files(file_live, file_staging):
        if file_live.name.endswith('.csv'):
            df_live = read_csv_file_with_detected_encoding(file_live, "str")
        else:
            df_live = read_excel_file(file_live, "str")

        if file_staging.name.endswith('.csv'):
            df_staging = read_csv_file_with_detected_encoding(file_staging, "str")
        else:
            df_staging = read_excel_file(file_staging, "str")

        if df_live.empty or df_staging.empty:
            show_warning_message("Warning: One or both of the uploaded files are empty.")
            return None, None
        else:
            return df_live, df_staging
    return None, None


# Data Matching and Analysis -------------------------------------------------------------------------------------------

def get_sbert_embeddings(text_list, multilingual=False):
    model_name = 'distiluse-base-multilingual-cased-v2' if multilingual else 'all-MiniLM-L6-v2'
    sbert_model = SentenceTransformer(model_name)
    embeddings = sbert_model.encode(text_list, show_progress_bar=True)
    return embeddings

def match_columns_and_compute_scores(model, df_live, df_staging, matching_columns, progress_bar, selected_model):
    matches_scores = {}
    total_columns = len(matching_columns)

    for i, col in enumerate(matching_columns):
        if col in df_live.columns and col in df_staging.columns:
            live_list = df_live[col].fillna('').tolist()
            staging_list = df_staging[col].fillna('').tolist()

            if model == "SBERT & Cosine Similarity":
                multilingual = "Multi-Lingual" in selected_model
                live_embeddings = get_sbert_embeddings(live_list, multilingual=multilingual)
                staging_embeddings = get_sbert_embeddings(staging_list, multilingual=multilingual)

                live_embeddings = np.array(live_embeddings)
                staging_embeddings = np.array(staging_embeddings)

                live_embeddings_norm = live_embeddings / np.linalg.norm(live_embeddings, axis=1, keepdims=True)
                staging_embeddings_norm = staging_embeddings / np.linalg.norm(staging_embeddings, axis=1, keepdims=True)

                cosine_similarities = np.dot(live_embeddings_norm, staging_embeddings_norm.T)

                matches = pd.DataFrame(columns=['From', 'To', 'Similarity'])
                for idx in range(len(live_list)):
                    cos_scores = cosine_similarities[idx]
                    max_idx = cos_scores.argmax()
                    max_score = cos_scores[max_idx]
                    match = pd.DataFrame({
                        'From': [live_list[idx]],
                        'To': [staging_list[max_idx]],
                        'Similarity': [max_score]
                    })
                    matches = pd.concat([matches, match], ignore_index=True)

                matches_scores[col] = matches
            else:
                model.match(live_list, staging_list)
                matches = model.get_matches()
                matches_scores[col] = matches

            progress = (i + 1) / total_columns
            if progress > 1:
                progress = 1
            progress_bar.progress(progress)
        else:
            st.warning(f"The column '{col}' does not exist in both the live and staging data.")

    return matches_scores

def setup_matching_model(selected_model):
    if "All-Mini-LM-v6" in selected_model or "Multi-Lingual" in selected_model:
        return "SBERT & Cosine Similarity"
    elif selected_model == "Edit Distance":
        return PolyFuzz(EditDistance())
    elif selected_model == "RapidFuzz":
        return PolyFuzz(RapidFuzz())
    else:
        return PolyFuzz(TFIDF(min_similarity=0))
    
def identify_best_matching_url(row, matches_scores, matching_columns, df_staging):
    # The first column chosen by the user acts as the primary target
    primary_col = matching_columns[0] 
    
    best_match_info = {'Best Match on': None, f'Highest Matching {primary_col}': None,
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
                    # Look up the matching staging value using the primary_col
                    staging_primary_val = df_staging.loc[df_staging[col] == match_row.iloc[0]['To'], primary_col].values
                    
                    best_match_info.update({
                        'Best Match on': col,
                        f'Highest Matching {primary_col}': staging_primary_val[0] if staging_primary_val.size > 0 else None,
                        'Highest Similarity Score': similarity_score,
                        'Best Match Content': match_row.iloc[0]['To']
                    })

    best_match_info['Median Match Score'] = np.median(similarities) if similarities else None
    return best_match_info, similarities


def add_additional_info_to_match_results(best_match_info, df_staging, matching_columns):
    primary_col = matching_columns[0]
    additional_columns = matching_columns[1:]
    
    for additional_col in additional_columns:
        if additional_col in df_staging.columns:
            # Look up the additional column values in staging based on our primary matching row
            staging_value = df_staging.loc[
                df_staging[primary_col] == best_match_info[f'Highest Matching {primary_col}'], additional_col].values
            best_match_info[f'Staging {additional_col}'] = staging_value[0] if staging_value.size > 0 else None
            
    return best_match_info


def identify_best_matching_url_and_median(df_live, df_staging, matches_scores, matching_columns):
    def process_row(row):
        best_match_info, similarities = identify_best_matching_url(row, matches_scores, matching_columns, df_staging)
        best_match_info = add_additional_info_to_match_results(best_match_info, df_staging, matching_columns)
        best_match_info['All Column Match Scores'] = [
            (col, f"{round(score * 100)}%" if not pd.isna(score) else "NaN%")
            for col, score in zip(matching_columns, similarities)
        ]
        return pd.Series(best_match_info)

    return df_live.apply(process_row, axis=1)


def finalise_match_results_processing(df_live, df_staging, matches_scores, matching_columns):
    match_results = identify_best_matching_url_and_median(df_live, df_staging, matches_scores, matching_columns)
    df_final = prepare_concatenated_dataframe_for_display(df_live, match_results, matching_columns)
    return df_final


def process_uploaded_files_and_match_data(df_live, df_staging, matching_columns, progress_bar, message_placeholder, selected_model):
    df_live = convert_dataframe_to_lowercase(df_live)
    df_staging = convert_dataframe_to_lowercase(df_staging)

    model = setup_matching_model(selected_model)
    matches_scores = process_column_matches_and_scores(model, df_live, df_staging, matching_columns, progress_bar, selected_model)

    message_placeholder.info('Finalising the processing. Please Wait!')
    df_final = finalise_match_results_processing(df_live, df_staging, matches_scores, matching_columns)

    display_final_results_and_download_link(df_final, 'migration_mapping_data.xlsx')
    message_placeholder.success('Complete!')

    return df_final


def generate_score_distribution_dataframe(df_final):
    df_final['Median Match Score Scaled'] = df_final['Median Match Score'] * 100
    bins = range(0, 110, 10)
    labels = [f'{i}-{i + 10}' for i in range(0, 100, 10)]
    df_final['Score Bracket'] = pd.cut(df_final['Median Match Score Scaled'], bins=bins, labels=labels, include_lowest=True)
    score_brackets = df_final['Score Bracket'].value_counts().sort_index().reindex(labels, fill_value=0)

    score_data = pd.DataFrame({
        'Score Bracket': score_brackets.index,
        'URL Count': score_brackets.values
    })
    return score_data


def select_columns_for_matching(df_live, df_staging):
    common_columns = list(set(df_live.columns) & set(df_staging.columns))
    
    st.write("Select up to 4 columns to match. **The first column you select will act as your primary identifier.**")
    
    if not common_columns:
        st.info("No common columns found between the files.")
        return []
        
    # Attempt to provide smart defaults based on common SEO exports
    defaults = ['Address', 'URL', 'url', 'slug', 'H1-1', 'Title 1']
    default_selection = [col for col in defaults if col in common_columns][:4]

    selected_columns = st.multiselect("Matching Columns", common_columns,
                                      default=default_selection,
                                      max_selections=4)
        
    return selected_columns


def process_column_matches_and_scores(model, df_live, df_staging, matching_columns, progress_bar, selected_model):
    return match_columns_and_compute_scores(model, df_live, df_staging, matching_columns, progress_bar, selected_model)


# Data Visualization and Reporting -------------------------------------------------------------------------------------

def display_final_results_and_download_link(df_final, filename):
    show_download_link_for_final_excel(df_final, filename)
    st.balloons()


# Excel File Operations ------------------------------------------------------------------------------------------------

def generate_excel_download_and_display_link(df, filename, score_data):
    output = io.BytesIO()
    excel_writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    df.to_excel(excel_writer, sheet_name='Mapped URLs', index=False)
    score_data.to_excel(excel_writer, sheet_name='Median Score Distribution', index=False)

    workbook = excel_writer.book
    worksheet1 = excel_writer.sheets['Mapped URLs']

    left_align_format = workbook.add_format({'align': 'left'})
    percentage_format = workbook.add_format({'num_format': '0.00%', 'align': 'center'})

    num_rows = len(df)
    num_cols = len(df.columns)
    worksheet1.add_table(0, 0, num_rows, num_cols - 1, {'columns': [{'header': col} for col in df.columns]})
    worksheet1.freeze_panes(1, 0)

    max_col_width = 80
    for i, col in enumerate(df.columns):
        col_width = max(len(col), max(df[col].astype(str).apply(len).max() if not df[col].empty else 0, 10)) + 2
        col_width = min(col_width, max_col_width)

        if 'Score' in col or 'Similarity' in col:
            worksheet1.set_column(i, i, col_width, percentage_format)
            worksheet1.conditional_format(1, i, num_rows, i, {
                'type': '3_color_scale',
                'min_color': "#f8696b",  
                'mid_color': "#ffeb84",  
                'max_color': "#63be7b"   
            })
        else:
            worksheet1.set_column(i, i, col_width, left_align_format)

    worksheet2 = excel_writer.sheets['Median Score Distribution']
    chart = workbook.add_chart({'type': 'column'})
    max_row = len(score_data) + 1

    chart.add_series({
        'name': "='Median Score Distribution'!$B$1",
        'categories': "='Median Score Distribution'!$A$2:$A$" + str(max_row),
        'values': "='Median Score Distribution'!$B$2:$B$" + str(max_row),
    })

    chart.set_title({'name': 'Distribution of Median Match Scores'})
    chart.set_x_axis({'name': 'Median Match Score Brackets'})
    chart.set_y_axis({'name': 'URL Count'})
    worksheet2.insert_chart('D2', chart)

    excel_writer.close()
    
    output.seek(0)
    st.download_button(
        label=f"📥 Download Mapping Data Excel File",
        data=output,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def show_download_link_for_final_excel(df_final, filename):
    df_for_score_data = df_final.drop(['Median Match Score Scaled', 'Score Bracket'], axis=1, inplace=False, errors='ignore')
    score_data = generate_score_distribution_dataframe(df_for_score_data)
    generate_excel_download_and_display_link(df_final, filename, score_data)


# Main Function and Additional Utilities -------------------------------------------------------------------------------

def format_match_scores_as_strings(df):
    df['All Column Match Scores'] = df['All Column Match Scores'].apply(lambda x: str(x) if x is not None else None)
    return df


def merge_live_and_matched_dataframes(df_live, match_results, matching_columns):
    # Only bring over the columns the user actually selected to keep the final output clean
    return pd.concat([df_live[matching_columns], match_results], axis=1)


def prepare_concatenated_dataframe_for_display(df_live, match_results, matching_columns):
    final_df = merge_live_and_matched_dataframes(df_live, match_results, matching_columns)
    final_df = format_match_scores_as_strings(final_df)
    return final_df


def main():
    setup_streamlit_interface()

    with st.expander("Advanced Settings"):
        model_options = [
            "Default (All-Mini-LM-v6)",
            "Multi-Lingual (distiluse-base-multilingual-cased-v2)",
            "TF-IDF"
        ]
        selected_model = st.selectbox("Select Matching Model", model_options)

        if selected_model == "TF-IDF":
            st.write("Use TF-IDF for comprehensive text analysis, suitable for high numbers of URLs (10K+) where SBERT may run into resourcing issues.")
        elif "All-Mini-LM-v6" in selected_model:
            st.write("Use SBERT for semantic matching of URLs, achieving a high success rate. Limit input to ~1,000 URLs to avoid memory issues.")
        elif "Multi-Lingual" in selected_model:
            st.write("Use Multilingual SBERT to compare across languages. Best for international content migrations.")

    file_live, file_staging = create_file_uploader_widgets()
    if file_live and file_staging:
        df_live, df_staging = process_and_validate_uploaded_files(file_live, file_staging)
        if df_live is not None and df_staging is not None:
            selected_columns = select_columns_for_matching(df_live, df_staging)
            
            # Ensure the user has selected at least one column before allowing processing
            if st.button("Process Files"):
                if not selected_columns:
                    st.warning("Please select at least one column to match.")
                else:
                    df_final = handle_data_matching_and_processing(df_live, df_staging, selected_columns, selected_model)

    create_page_footer_with_contact_info()

if __name__ == "__main__":
    main()
