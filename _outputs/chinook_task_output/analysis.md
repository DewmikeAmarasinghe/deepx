Summary of findings and recommendations

Key findings

- Top genres: Rock is the clear leader by revenue (total ≈ 826.65), followed by Latin (≈ 382.14), Metal (≈ 261.36), Alternative & Punk (≈ 241.56) and TV Shows (≈ 93.53). These five genres (with ties handled) account for a large share of total sales.

- Top customers: For Rock, highest-spending customers include Eduardo Martins (Brazil, 28.71), Robert Brown (Canada, 24.75) and multiple customers tied at ≈21.78. For Latin, notable top customers are Alexandre Rocha and Roberto Almeida (both Brazil, ≈15.84). For Metal and Alternative & Punk, top individual customers contribute meaningful single-customer revenue (e.g., Hannah Schneider for Metal, Daan Peeters for Alternative & Punk).

- Monthly series and geography: Revenue is well distributed across many countries and months; Rock shows the most consistent and highest revenue across many countries and months. Latin and Metal show strong pockets of demand by country and month (for example, Latin spikes in USA in 2022-05 and Brazil across multiple months).

- Spikes and declines: The month-over-month spike/decline detection flagged many genre-country-months where revenue moved by 50% or more vs the prior month. Examples include:
  - Rock (USA): 2025-10 jumped dramatically vs prior month (1300% in the detected data — large relative jump from a small prior month value). Many Rock country series show substantial volatility (large increases and large declines) across the multi-year span.
  - Latin (USA): 2022-05 shows a large spike (≈1300% relative increase) — this corresponds to a concentrated set of invoices in that month for that country/genre.
  - Metal (USA): 2025-07 shows a very large increase driven by concentrated transactions in that month.

Notes on percent-change handling

- Percent change is computed as (current - previous) / previous * 100 and rounded to two decimals. If previous revenue is NULL (no prior month for that genre-country) the percent change is NULL and the row is not used for spike detection.
- If previous revenue equals zero we treat percent change as NULL for numeric reporting but we include the row as a flagged spike only if previous=0 and current>0. This handles the practical case where a new revenue event after zero should be treated as a significant spike. This decision is noted in the results and in the code comments.

Possible drivers and interpretation

- Catalog promotions or marketing campaigns: Large localized spikes (for a single country and month) often point to targeted promotions or marketing campaigns, clearance bundles, or playlist placements that temporarily drive purchases in specific markets.
- New releases or featured content: For TV Shows or Soundtrack spikes, a concentration of purchases in a month could reflect a newly released season or soundtrack, or placement in a curated editorial list.
- Data sparsity and small-denominator effects: Many large percent changes stem from very small prior-month revenue (e.g., previous month 0.99 or 0.99 to 13.86) — the relative percent moves appear large even when absolute values are modest. Interpret percent-change flags alongside absolute revenue values to prioritize material events.
- Country-specific adoption and seasonality: Some countries show recurring seasonality (e.g., higher Rock activity in summer months for certain countries). Others show one-off events.

Recommendations

1) Prioritize follow-up on high-absolute-impact flags: Filter the spikes table for both pct_change and absolute revenue (e.g., revenue > $5 or >$10) to surface events that materially impact revenue rather than those driven by very small prior-month bases.

2) Investigate marketing and release calendars for months with large, localized spikes (examples: Rock USA 2025-10, Latin USA 2022-05, Metal USA 2025-07). Correlate with campaign logs, playlist placements, and release dates.

3) For repeating pockets of strength (genres and countries that repeatedly show large revenue), consider allocating promotional budget or localized curation, and explore bundling or region-specific offers.

4) Improve monitoring: Add automated alerts that combine absolute thresholds and percent-change thresholds to reduce noise from small-base fluctuations.

5) Data hygiene: Ensure InvoiceDate timezone and billing country mapping are consistent; interpret multi-country customers carefully (billing country used here).

Files produced

- queries.sql — SQL used for analysis
- genre_month_country_revenue.csv — monthly revenue by genre and customer country
- top5_genres.csv — top genres (with ties)
- top_customers_per_genre.csv — top 5 customers for each top genre
- spikes_and_declines.csv — flagged month-over-month changes >=50% (or prior=0→current>0)

