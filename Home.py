__import__("pysqlite3") 
import sys 
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from utils.layout import page_config
from utils.ai_inference import gpt4o_inference_with_search, gpt4o_inference
from utils.chroma_db import initialise_persistent_chromadb_client_and_collection, add_document_chunk_to_chroma_collection, query_chromadb_collection

page_config()

if "log" not in st.session_state:
    st.session_state.log = ""

if "query" not in st.session_state:
    st.session_state.query = None

if "report" not in st.session_state:
    st.session_state.report = None

if "collection" not in st.session_state: 

    st.session_state.collection = initialise_persistent_chromadb_client_and_collection("dd_documents")

if "number_updates" not in st.session_state:

    st.session_state.number_updates = 0

def summary_agent(brief, report):

    SYSTEM_PROMPT = "You are a lawyer, expert at managing and completing corporate due diligence."

    INSTRUCTION = f"You have been briefed on the following:\n<brief>{brief}</brief>. You have the following draft report:<draft_report>{report}</draft_report>. Using the draft report, create a highly professional, final due diligence report to send to the client in response to the brief. The final report should only contain information that directly responds to the brief."

    final_report = gpt4o_inference(SYSTEM_PROMPT, INSTRUCTION)

    return final_report

def search_agent(instruction):

    SYSTEM_PROMPT = "You are a legal document processing assistant, specialising in searching vector databases to find relevant information."

    INSTRUCTION = f"You have been instructed to create a search for the following information, and return any relevant documents.\n<search_requirement>{instruction}</search_requirement>."

    search_results = gpt4o_inference_with_search(SYSTEM_PROMPT, INSTRUCTION)

    return search_results

def lawyer_agent(brief, report=""):

    if st.session_state.number_updates == 5:

        st.markdown("Report Finalised")

        final_report = summary_agent(brief, report)

        return final_report

    SYSTEM_PROMPT = "You are a lawyer, expert at managing and completing corporate due diligence."

    INSTRUCTION_1 = f"You have been briefed on the following:\n<brief>{brief}</brief>. You have the following information so far in your report:<report>{report}</report>. You have an assistant who is capable of searching a dataroom of documents using a vector database query. Come up with an instruction to your assistant to find new documents."

    search_instruction = gpt4o_inference(SYSTEM_PROMPT, INSTRUCTION_1)

    st.markdown("Briefing Search Agent")

    st.session_state.log += f"""
    ## SEARCH INSTRUCTION
    {search_instruction}
    \n\n
    """

    new_documents = search_agent(search_instruction)

    st.markdown("Reviewing Documents")

    st.session_state.log += f"""
    ## SEARCH RESULTS
    {new_documents}
    \n\n
    """

    INSTRUCTION_2 = f"You are conducting an M&A due diligence and preparing a report to your client. There are thousands of documents in a dataroom. You have been briefed on the following:\n<brief>{brief}</brief>. You have the following information so far in your report:<report>{report}</report>. You have received the following new documents from your assistant from the dataroom:\n<new_documents>{new_documents}</new_documents>. Use the new documents to help draft your report, responding to the brief. Your report should be formatted in markdown.<important>If you do not think it is necessary to add further detail to your report, or to instruct your assistance to look for further documents in the dataroom, you must also include the word 'STOP' at the end of your analysis.</important>"

    response = gpt4o_inference(SYSTEM_PROMPT, INSTRUCTION_2)

    st.markdown("Drafting Report")

    st.session_state.log += f"""
    ## LAWYER RESPONSE
    {response}
    \n\n
    """

    if "STOP" in response.upper() and st.session_state.number_updates > 1:

        report += response

        st.markdown("Report Finalised")

        final_report = summary_agent(brief, report)

        return final_report
    
    else:

        report += response

        st.markdown("Updating Report")

        st.session_state.number_updates = st.session_state.number_updates + 1

        return lawyer_agent(brief, report)

if st.session_state.query == None or st.session_state.query == "":

    st.markdown("## Brief")
    st.session_state.query = st.text_area(
        label="query",
        label_visibility="collapsed"
    )

if st.button("Run Brief"):

    st.session_state.report = lawyer_agent(st.session_state.query)
    st.session_state.number_updates = 0

if st.session_state.report is not None:

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("## REPORT")

        with st.container(border=True):

            st.markdown(st.session_state.report)
    
    with col2:

        st.markdown("## LOG")

        with st.container(border=True):

            st.markdown(st.session_state.log)

