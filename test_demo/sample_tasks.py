ORCHESTRATOR_TASKS: list[str] = [
    """\
You are a capable agent like openclaw. I want you to register a new skill for yourself. Fetch https://lobehub.com/skills/openclaw-openclaw-weather/skill.md and save it in 
_outputs/weather_task_output/skills/weather/SKILL.md.
Then edit your web agent definition in test_demo/web_agent.py file to include the new skill.
Then use that new skill to find the current weather of colombo, Sri Lanka.
Save your outputs to the _outputs/weather_task_output folder.
""",
    """\
Using the **skill-creator** guidance, add a minimal new skill bundle at **test_demo/skills/sample-rerun/SKILL.md** with frontmatter `name` and `description` and a short "When to use" section.
Do not modify unrelated agents. Return the path you created and one sentence on when to use it.
""",
    """\
Using the **cron-scheduling** skill, draft a crontab entry that runs daily at 02:30 (explain timezone assumptions). Save the snippet and a short explanation to **/_outputs/cron_sample.txt**.
Return that path and a one-line summary.
""",
    """\
Combine **cron-scheduling** and **skill-creator**: document a daily housekeeping job (what it runs, log path idea) in **/_outputs/daily_job_notes.md**, then create **test_demo/skills/daily-housekeeping/SKILL.md** that points to those notes and summarizes the schedule.
Return both paths.
""",
]

WEB_AGENT_TASKS: list[str] = [
    """\
https://www.parliament.lk/en  Find the directory of current MPs. Go to their profiles and finally create a table showing name, party, phone number, date of birth, religion. The table should have 
only 50 such profile inforamtion. Number the table rows.
Save your outputs to the _outputs/parliament_task_output folder.
""",
    """\
Go to the blog site https://thamalu.blogspot.com/ and find the 5 most top read articles go to each link and then create a summary of each article.. Finally create create a report showing the articles and their summaries.. Then create an overall summary of the blog.
Save your outputs to the _outputs/blog_task_output folder.
""",
    """\
I want a clear, well-sourced picture of sodium-ion versus lithium-ion for electric vehicles—
energy density limits, materials and geopolitics, manufacturing scale-up, then write a single strong markdown report in the workspace
I can open, and show me that report in the terminal when it is ready.
Save your outputs to the _outputs/sodium_ion_task_output folder.
""",
    """\
I want to get a table of ENTC department academic staff of university of Moratuwa. In the table show Name, Highest Degree, The institution that offered the highest degree, residential telephone number.
Save your outputs to the _outputs/entc_staff_task_output folder.
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

Save your outputs to the _outputs/chinook_task_output folder.
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

Save your output to the _outputs/northwind_task_output folder.
""",
    """\
The Chinook database (**chinook.db**) is available under **test_demo/dbs/test_dbs**.

Answer: Which **genre** has the highest total invoice line **quantity** sold, and what is that total quantity? Return the SQL, a small result table, and one sentence of interpretation.

Save your outputs to **/_outputs/chinook_genre_quantity_task/** (SQL and brief notes as needed).
""",
]

PDF_AGENT_TASKS: list[str] = [
    """\
I have two pdf files in the test_demo/pdfs folder. They are attention.pdf and gpt4.pdf.
Perform a deep analysis of each document and produce a structured output for both.
 
For each paper, provide a detailed breakdown of:
- Core ideas and contributions
- Architecture and design choices (with emphasis on technical structure)
- Training methodology and data usage (if present)
- Evaluation setup, benchmarks, and performance metrics
- Key numerical results and tables
- Limitations, weaknesses, and assumptions
 
Then perform a comparative analysis focusing on:
- How core ideas and model paradigms evolve from attention-based models to GPT-4 level systems
- Architectural and methodological changes across the two papers
- Differences in scaling strategies, performance improvements, and design philosophy
- Key innovations introduced and what was replaced or improved over time

- You must not convert each page to an image.
- You should convert the entire pdfs to singe text files and read them.
 
Finally, produce:
- A combined detailed research report synthesizing both papers
- A merged PDF containing both documents in a unified format for reference
 
Save your output to the _outputs/pdf_task_output folder.
""",
]

HF_AGENT_TASKS: list[str] = [
    """\
You have a huggingface sub agent with Hugging Face Hub tools (when HF_TOKEN is set).

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

Save your output to the _outputs/hf_task_output folder.
"""
]
