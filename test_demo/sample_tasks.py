"""Sample task strings for manual testing (e.g. copy into the orchestrator CLI)."""

SAMPLE_TASKS: list[str] = [
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
Search Hugging Face Hub via MCP (hf_agent): shortlist three recent diffusion-model papers with
links and one-line relevance; save a workspace digest path and summarise trade-offs for practitioners.
Requires HF_TOKEN and Node/npx for the MCP server.
""",
    """\
From the Chinook sample DB: which three genres have the most tracks, and within each genre who
are the top three artists by track count? Show SQL and tables; then a short executive readout.
""",
    """\
Northwind-style retail: each customer's order count, total quantity, and average order value,
ranked. I need SQL, results, and one paragraph on what would break in a messy production schema.
""",
    """\
I have two research PDFs under /test_demo/pdfs/ (attention.pdf and gpt4.pdf). Summarize each
in a structured way (ideas, architectures, limitations), compare how themes evolved, extract any
key tables or numbers you can, then produce one combined workspace report and a merged PDF—
return paths and render the report.
""",
    """\
arXiv: shortlist recent papers that materially change efficient LLM inference assumptions this
quarter; for each, title, why it matters, and a link. Write paths to a workspace digest and
render it.
""",
    """\
Run a short non-interactive shell check: `uname -a` and `python3 -c "import sys; print(sys.version)"`.
Paste both outputs in your reply and say whether the shell looks healthy.
""",
    """\
Curl https://lobehub.com/skills/openclaw-openclaw-weather/skill.md, then follow the instructions
to set up LobeHub Skills Marketplace and install the skill in test_demo/skills folder. Once installed, read its SKILL.md file
in the installed directory and follow its instructions to complete the task.
""",
]