import asyncio
import json
import os
import sys

from dotenv import dotenv_values
from google import genai

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# LOAD API KEY
# ============================================================

config = dotenv_values(".env")

api_key = config.get("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY not found!")
    sys.exit(1)


# ============================================================
# GEMINI
# ============================================================

gemini = genai.Client(
    api_key=api_key
)


# ============================================================
# RAG DATABASE
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="vector_db"
)

collection = chroma_client.get_collection(
    name="company_documents"
)


# ============================================================
# RAG FUNCTION
# ============================================================

def search_company_documents(question):

    # --------------------------------------------------------
    # Get documents from Chroma
    # --------------------------------------------------------

    data = collection.get(
        include=["documents"]
    )

    documents = data.get("documents", [])

    if not documents:
        return "No company documents found."

    # --------------------------------------------------------
    # Create the SAME TF-IDF model used for the database
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer()

    document_embeddings = vectorizer.fit_transform(
        documents
    ).toarray()

    # --------------------------------------------------------
    # Convert user question into TF-IDF vector
    # --------------------------------------------------------

    question_embedding = vectorizer.transform(
        [question]
    ).toarray()[0]

    # --------------------------------------------------------
    # Search Chroma using our own embedding
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=3
    )

    retrieved_documents = results.get(
        "documents",
        []
    )

    if not retrieved_documents:
        return "No relevant company documents found."

    # --------------------------------------------------------
    # Flatten results
    # --------------------------------------------------------

    flattened = []

    for group in retrieved_documents:

        for document in group:

            flattened.append(document)

    return "\n\n".join(flattened)


# ============================================================
# RAG TOOL FOR GEMINI
# ============================================================

rag_tool = {
    "type": "function",
    "name": "search_company_documents",
    "description": (
        "Search the company handbook and internal company "
        "documents for policies, employee information, "
        "working hours, leave, work from home, training, "
        "performance and other company-related information."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The company-related question to search for."
                )
            }
        },
        "required": ["question"]
    }
}


# ============================================================
# MCP SERVER
# ============================================================

server_params = StdioServerParameters(
    command=sys.executable,
    args=[
        "mcp_server/server.py"
    ],
)


# ============================================================
# CONVERT MCP TO GEMINI TOOLS
# ============================================================

def convert_mcp_tools_to_gemini(mcp_tools):

    tools = []

    for tool in mcp_tools:

        tools.append({
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": (
                tool.inputSchema
                if tool.inputSchema
                else {
                    "type": "object",
                    "properties": {}
                }
            )
        })

    return tools


# ============================================================
# MCP RESULT
# ============================================================

def extract_mcp_result(result):

    output = []

    if hasattr(result, "content"):

        for item in result.content:

            if hasattr(item, "text"):

                output.append(item.text)

    if hasattr(result, "structuredContent"):

        if result.structuredContent:

            output.append(
                json.dumps(
                    result.structuredContent,
                    ensure_ascii=False
                )
            )

    if not output:

        return "Tool executed successfully."

    return "\n".join(output)


# ============================================================
# MAIN
# ============================================================

async def main():

    print("\n======================================")
    print("       AI BUSINESS ANALYST")
    print("======================================")

    # --------------------------------------------------------
    # CONNECT MCP
    # --------------------------------------------------------

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            print("\n🟢 Connected to MCP Server")

            # ------------------------------------------------
            # GET MCP TOOLS
            # ------------------------------------------------

            tool_result = await session.list_tools()

            mcp_tools = tool_result.tools

            print("\nAvailable MCP Tools:")

            for tool in mcp_tools:

                print(
                    f"  🔧 {tool.name}"
                )

            print(
                "  📚 search_company_documents"
            )

            # ------------------------------------------------
            # COMBINE MCP + RAG TOOLS
            # ------------------------------------------------

            gemini_tools = convert_mcp_tools_to_gemini(
                mcp_tools
            )

            gemini_tools.append(
                rag_tool
            )

            # ------------------------------------------------
            # USER QUESTION
            # ------------------------------------------------

            question = input(
                "\nAsk your Business Analyst: "
            )

            print(
                "\n🤖 Gemini is thinking..."
            )

            # ------------------------------------------------
            # FIRST GEMINI REQUEST
            # ------------------------------------------------

            interaction = gemini.interactions.create(

                model="gemini-3.6-flash",

                input=question,

                tools=gemini_tools
            )

            # ------------------------------------------------
            # FIND FUNCTION CALLS
            # ------------------------------------------------

            function_calls = []

            for step in interaction.steps:

                if step.type == "function_call":

                    function_calls.append(step)

                    print(
                        f"\n🔧 Gemini selected tool: "
                        f"{step.name}"
                    )

                    print(
                        f"📦 Arguments: "
                        f"{step.arguments}"
                    )

            # ------------------------------------------------
            # NO TOOL
            # ------------------------------------------------

            if not function_calls:

                print(
                    "\n======================================"
                )

                print(
                    "             AI ANSWER"
                )

                print(
                    "======================================"
                )

                print(
                    interaction.output_text
                )

                return

            # ------------------------------------------------
            # EXECUTE TOOLS
            # ------------------------------------------------

            function_results = []

            for call in function_calls:

                # ============================================
                # RAG TOOL
                # ============================================

                if call.name == "search_company_documents":

                    print(
                        "\n📚 Searching company documents..."
                    )

                    arguments = call.arguments or {}

                    rag_question = arguments.get(
                        "question",
                        question
                    )

                    result_text = search_company_documents(
                        rag_question
                    )

                    print(
                        f"\n📄 RAG Result:\n{result_text}"
                    )

                # ============================================
                # MCP TOOL
                # ============================================

                else:

                    print(
                        f"\n⚙️ Executing MCP tool: "
                        f"{call.name}"
                    )

                    arguments = call.arguments or {}

                    result = await session.call_tool(
                        call.name,
                        arguments
                    )

                    result_text = extract_mcp_result(
                        result
                    )

                    print(
                        f"\n📊 MCP Result:\n{result_text}"
                    )

                # ============================================
                # SEND RESULT TO GEMINI
                # ============================================

                function_results.append({

                    "type": "function_result",

                    "name": call.name,

                    "call_id": call.id,

                    "result": [
                        {
                            "type": "text",
                            "text": result_text
                        }
                    ]
                })

            # ------------------------------------------------
            # SEND RESULTS BACK TO GEMINI
            # ------------------------------------------------

            print(
                "\n🧠 Sending retrieved data to Gemini..."
            )

            final_interaction = (
                gemini.interactions.create(

                    model="gemini-3.6-flash",

                    previous_interaction_id=interaction.id,

                    input=function_results,

                    tools=gemini_tools
                )
            )

            # ------------------------------------------------
            # FINAL ANSWER
            # ------------------------------------------------

            print(
                "\n======================================"
            )

            print(
                "             AI ANSWER"
            )

            print(
                "======================================"
            )

            print(
                final_interaction.output_text
            )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())