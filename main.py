from dotenv import load_dotenv
import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# Load environment variables
load_dotenv()

# Set global variable for PDF path
path: Optional[str] = None

# Step 1: Configure the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

# Step 2: PDF Summarization Tool
def get_pdf_summarizer_tool(pdf_path: Optional[str]):
    @tool
    def pdf_summarizer(query: str) -> str:
        """
        Answers queries based on the PDF provided.
        """
        if not pdf_path:
            return "No PDF file provided. Please try using web search or Wikipedia."

        try:
            loader = PyPDFLoader(file_path=pdf_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)

            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embedding_model)

            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            results = retriever.invoke(query)

            return "\n\n".join([doc.page_content for doc in results])

        except Exception as e:
            return f"Error processing PDF: {e}"

    return pdf_summarizer

# Step 3: Web and Wikipedia tools
web_search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Step 4: Prompt and Agent Setup
prompt = hub.pull("hwchase17/react")

if __name__ == "__main__":
    # Step 5: Get user input
    query = input("Enter your query: ").strip()
    pdf_path = input("Enter PDF file path (optional): ").strip()
    if pdf_path == "":
        pdf_path = None

    # Step 6: Add tools dynamically
    tools = [web_search, wiki]
    if pdf_path:
        pdf_tool = get_pdf_summarizer_tool(pdf_path)
        tools.append(pdf_tool)

    # Step 7: Create and run agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    input_dict = {"input": query}
    response = agent_executor.invoke(input_dict)

    # Step 8: Output result
    print("\nAgent Response:\n")
    print(response["output"])