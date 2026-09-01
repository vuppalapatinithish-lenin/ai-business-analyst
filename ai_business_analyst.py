import asyncio
import json
import os
import sys

from dotenv import dotenv_values
from google import genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# LOAD API KEY
# ============================================================

config = dotenv_values(".env")

# Local .env OR Render Environment Variables
api_key = os.getenv("GEMINI_API_KEY") or config.get("GEMINI_API_KEY")

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
# RAG DATABASE - LIGHTWEIGHT TF-IDF
# ============================================================

DOCUMENT_PATH = os.path.join(
    os.path.dirname(__file__),
    "documents",
    "company_handbook.txt"
)


def load_company_documents():

    if not os.path.exists(DOCUMENT_PATH):
        return []

    with open(
        DOCUMENT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    # Split handbook into small chunks
    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    return chunks


document_chunks = load_company_documents()


# Create lightweight TF-IDF index
if document_chunks:

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    document_vectors = vectorizer.fit_transform(
        document_chunks
    )

else:

    vectorizer = None
    document_vectors = None


# ============================================================
# RAG FUNCTION
# ============================================================

def search_company_documents(question):

    if not document_chunks or vectorizer is None:
        return "No company documents found."

    question_vector = vectorizer.transform(
        [question]
    )

    similarities = cosine_similarity(
        question_vector,
        document_vectors
    )[0]

    # Get top 3 relevant chunks
    top_indices = similarities.argsort()[-3:][::-1]

    results = []

    for index in top_indices:

        if similarities[index] > 0:

            results.append(
                document_chunks[index]
            )

    if not results:
        return "No relevant company documents found."

    return "\n\n".join(results)


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

            print("\nConnected to MCP Server")

            # ------------------------------------------------
            # GET MCP TOOLS
            # ------------------------------------------------

            tool_result = await session.list_tools()

            mcp_tools = tool_result.tools

            print("\nAvailable MCP Tools:")

            for tool in mcp_tools:

                print(
                    f"  {tool.name}"
                )

            print(
                "  search_company_documents"
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
                "\nGemini is thinking..."
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
                        f"\nGemini selected tool: "
                        f"{step.name}"
                    )

                    print(
                        f"Arguments: "
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
                        "\nSearching company documents..."
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
                        f"\nRAG Result:\n{result_text}"
                    )

                # ============================================
                # MCP TOOL
                # ============================================

                else:

                    print(
                        f"\nExecuting MCP tool: "
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
                        f"\nMCP Result:\n{result_text}"
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
                "\nSending retrieved data to Gemini..."
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