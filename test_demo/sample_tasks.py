ORCHESTRATOR_TASKS: list[str] = [
    """\
Curl https://lobehub.com/skills/openclaw-openclaw-weather/skill.md, then follow the instructions
to set up LobeHub Skills Marketplace and install the skill in test_demo/skills folder. Once installed, read its SKILL.md file
in the installed directory and follow its instructions to find the weather of colombo, Sri Lanka now.
""",
]

WEB_AGENT_TASKS: list[str] = [
    """\
https://www.parliament.lk/en  Find the directory of current MPs. Go to their profiles and finally create a table showing name, party, phone number, date of birth, religion. The table should have 
only 30 such profile inforamtion. Number the table rows.
""",
    """\
Go to the blog site https://thamalu.blogspot.com/ and find the 5 most top read articles go to each link and then create a summary of each article.. Finally create create a report showing the articles and their summaries.. Then create an overall summary of the blog.
""",
    """\
I want a clear, well-sourced picture of sodium-ion versus lithium-ion for electric vehicles—
energy density limits, materials and geopolitics, manufacturing scale-up, then write a single strong markdown report in the workspace
I can open, and show me that report in the terminal when it is ready.
""",
    """\
arXiv: shortlist recent papers that materially change efficient LLM inference assumptions this
quarter; for each, title, why it matters, and a link. Write paths to a workspace digest and
render it.
""",
]

SQL_AGENT_TASKS: list[str] = [
    """\
The Chinook database contains information about customers, invoices,
invoice line items, tracks, albums, artists, genres, and employees.

A stakeholder reports that sales from high-value customers may be
concentrated in only a few music genres.

Investigate customer spending patterns by month, broken down by genre
and customer country. Identify the top five revenue-generating genres,
the customers contributing most to each, and any months with unusual
spending spikes or declines.

Return SQL, result tables, and a short business analysis explaining
possible drivers behind the trends.
""",
    """\
The Northwind database contains information about customers, employees,
orders, order details, products, categories, suppliers, and shippers.

A stakeholder reports that revenue dropped sharply during 1998.

Investigate monthly revenue trends for 1997–1998, broken down by
product category and shipping country. Identify the largest
month-over-month declines, the customers and product categories
contributing most to the drop, and any operational patterns that
may explain the change.

Return SQL, result tables, and a short root-cause analysis.
""",
]

PDF_AGENT_TASKS: list[str] = [
    """\
I have two research PDFs under /test_demo/pdfs/ (attention.pdf and gpt4.pdf). Summarize each
in a structured way (ideas, architectures, limitations), compare how themes evolved, extract any
key tables or numbers you can, then produce one combined workspace report and a merged PDF—
return paths and render the report.
""",
]

HF_AGENT_TASKS: list[str] = [
        """\
You have a huggingface sub agent who is connected to the Hugging Face Hub via MCP.

Search the Hugging Face Hub for three recent diffusion-model papers or official model cards.
Choose items that are practically useful for ML engineers and preferably from 2025–2026.

For each item, provide:
- title
- link
- one-line explanation of why it is relevant

Then compare the three options in a short summary focusing on:
- quality
- inference speed
- compute requirements
- best use case

Finally, save a short digest to:
_outputs/diffusion_models_summary.md

Return the final shortlist, the saved workspace path, and a brief recommendation on which model a practitioner should evaluate first.
"""
]