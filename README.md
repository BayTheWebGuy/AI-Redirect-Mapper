# AI-Redirect-Mapper

How to Use
Prepare Data:

Crawl both the live and staging websites using Screaming Frog SEO Spider.
Export the crawled data as CSV files.
Upload Files:

Use the tool's interface to upload the CSV files for the live and staging websites.
Column Selection:

By default, the application searches for columns named 'Address', 'H1-1', and 'Title 1'. These can be manually mapped if not automatically found.
Users can select up to three columns for the matching process.
Processing:

Click the 'Process Files' button to start the comparison and matching process.
Download Results:

Once the processing is complete, a link to download the output file will be provided.
Installation
To run this application locally, you'll need Python and Streamlit installed. Clone the repository, navigate to the app's directory, and run it using Streamlit:

streamlit run website-migration.py
