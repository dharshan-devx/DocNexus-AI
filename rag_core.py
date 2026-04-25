import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

class RAGCore:
    def __init__(self, es_url="http://localhost:9200", index_name="summaries_index"):
        self.es_url = es_url
        self.index_name = index_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = ElasticsearchStore(
            embedding=self.embeddings,
            es_url=self.es_url,
            index_name=self.index_name,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy()
        )
        self.model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

    def get_available_reports(self) -> List[str]:
        # A simple hack to get available reports: ideally we'd query ES aggregations
        # For now, we will assume standard names, but in a real app you'd do an ES terms aggregation on metadata.pdf_title
        # To keep it robust without writing complex ES queries, let's just return a placeholder or dynamically fetch
        # Since we use langchain, we can do a dummy search to get unique titles
        try:
            results = self.vectorstore.client.search(
                index=self.index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "pdf_titles": {
                            "terms": {"field": "metadata.pdf_title.keyword", "size": 100}
                        }
                    }
                }
            )
            buckets = results.get("aggregations", {}).get("pdf_titles", {}).get("buckets", [])
            return [b["key"] for b in buckets]
        except Exception as e:
            print(f"Error fetching reports: {e}")
            return []

    def process_query(self, user_query: str, chat_history: List[Any]):
        available_reports = self.get_available_reports()
        reports_str = "\n".join(available_reports) if available_reports else "Any available report"
        
        # Step 1: Contextualize the query based on chat history
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_query}"),
            ]
        )
        contextualize_chain = contextualize_q_prompt | self.model | StrOutputParser()
        
        standalone_query = contextualize_chain.invoke({"user_query": user_query, "chat_history": chat_history})
        print(f"Standalone Query: {standalone_query}")

        # Step 2: Determine which PDFs to query
        get_pdf_query = f"""You are an assistant tasked with generating additional questions from the given query. \
Given a set of questions, give the relevant questions (in the format as shown) pertaining to each individual company \
in the query IF there are more than one. Also give the report name it corresponds to.
Available Report names:
{reports_str}

<--example start-->
Query: What are the equity compensation plans of AMD and Cisco?
Answer:
What are the equity compensation plans of AMD?, AMD.10K.2023.pdf
What are the equity compensation plans of Cisco?, CSCO.10K.2023.pdf
<--example end-->

<--example start-->
Query: Are there any ongoing legal disputes with Uber?
Answer:
Are there any ongoing legal disputes with Uber?, UBER.10K.2023.pdf
<--example end-->

Query: {{standalone_query}}
Answer:
"""
        get_pdf_query_prompt = ChatPromptTemplate.from_template(get_pdf_query)
        get_pdf_query_chain = get_pdf_query_prompt | self.model | StrOutputParser()
        pdf_queries_str = get_pdf_query_chain.invoke({"standalone_query": standalone_query})
        
        # Step 3: Retrieve context
        contexts = []
        for line in pdf_queries_str.split('\n'):
            line = line.strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                q = parts[0].strip()
                pdf = parts[1].strip()
                
                # Retrieve from VectorStore
                docs = self.vectorstore.similarity_search(
                    q, 
                    k=4, 
                    filter=[{"term": {"metadata.pdf_title.keyword": pdf}}]
                )
                contexts.extend(docs)

        # Parse contexts
        context_str = ""
        for i, doc in enumerate(contexts):
            title = doc.metadata.get('pdf_title', 'Unknown Source')
            context_str += f"--- CONTEXT {i+1} [Source: {title}] ---\n{doc.page_content}\n\n"

        # Step 4: Generate Answer
        rag_prompt_text = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question \
in as many words as required. Feel free to go into the details of what's presented in the context down below.
If you don't know the answer, just say "I don't know."
ALWAYS cite your sources using the [Source: ...] format provided in the context blocks when making factual claims.

Context:
{context}

Question: {question}
Answer: 
"""
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", rag_prompt_text),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # We return the runnable so Streamlit can stream it
        rag_chain = rag_prompt | self.model | StrOutputParser()
        
        return rag_chain.stream({
            "context": context_str,
            "chat_history": chat_history,
            "question": standalone_query
        }), context_str
