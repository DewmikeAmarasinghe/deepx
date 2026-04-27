-- Queries used for Chinook analysis

-- 1) genre_month_country_revenue
-- Aggregated monthly revenue (YYYY-MM) by genre and customer country
SELECT strftime('%Y-%m', i.InvoiceDate) AS year_month,
       g.GenreId AS genre_id,
       g.Name AS genre_name,
       c.Country AS country,
       ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue
FROM Invoice i
JOIN Customer c ON i.CustomerId = c.CustomerId
JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
JOIN Track t ON il.TrackId = t.TrackId
JOIN Genre g ON t.GenreId = g.GenreId
GROUP BY year_month, genre_id, country
ORDER BY year_month, genre_name, country;

-- 2) top genres (include ties) — rank by total revenue across all time
SELECT rank_info.rnk AS rank, rank_info.genre_id, rank_info.genre_name, ROUND(rank_info.total_revenue,2) AS total_revenue
FROM (
  SELECT g.GenreId AS genre_id, g.Name AS genre_name, SUM(il.UnitPrice*il.Quantity) AS total_revenue,
         RANK() OVER (ORDER BY SUM(il.UnitPrice*il.Quantity) DESC) AS rnk
  FROM Invoice i
  JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
  JOIN Track t ON il.TrackId = t.TrackId
  JOIN Genre g ON t.GenreId = g.GenreId
  GROUP BY g.GenreId
) AS rank_info
WHERE rank_info.rnk <= 5
ORDER BY rank_info.rnk, rank_info.total_revenue DESC;

-- 3) top 5 customers per top genres (top genres determined by previous query)
WITH genre_list AS (
  SELECT genre_id FROM (
    SELECT g.GenreId AS genre_id, SUM(il.UnitPrice*il.Quantity) AS total_revenue,
           RANK() OVER (ORDER BY SUM(il.UnitPrice*il.Quantity) DESC) AS rnk
    FROM Invoice i
    JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
    JOIN Track t ON il.TrackId = t.TrackId
    JOIN Genre g ON t.GenreId = g.GenreId
    GROUP BY g.GenreId
  ) WHERE rnk <= 5
), customer_genre_totals AS (
  SELECT g.GenreId AS genre_id, g.Name AS genre_name, c.CustomerId AS customer_id, c.FirstName AS customer_first_name, c.LastName AS customer_last_name, c.Country AS country,
         SUM(il.UnitPrice * il.Quantity) AS customer_total_revenue
  FROM Invoice i
  JOIN Customer c ON i.CustomerId = c.CustomerId
  JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
  JOIN Track t ON il.TrackId = t.TrackId
  JOIN Genre g ON t.GenreId = g.GenreId
  WHERE g.GenreId IN (SELECT genre_id FROM genre_list)
  GROUP BY g.GenreId, c.CustomerId
)
SELECT genre_id, genre_name, customer_id, customer_first_name, customer_last_name, country, ROUND(customer_total_revenue,2) AS customer_total_revenue,
       rnk
FROM (
  SELECT *, RANK() OVER (PARTITION BY genre_id ORDER BY customer_total_revenue DESC) AS rnk
  FROM customer_genre_totals
) x
WHERE rnk <= 5
ORDER BY genre_id, rnk, customer_total_revenue DESC;

-- 4) spikes and declines (month-over-month percent change per genre-country series)
WITH gm AS (
  SELECT strftime('%Y-%m', i.InvoiceDate) AS year_month,
         g.GenreId AS genre_id,
         g.Name AS genre_name,
         c.Country AS country,
         ROUND(SUM(il.UnitPrice * il.Quantity),2) AS revenue
  FROM Invoice i
  JOIN Customer c ON i.CustomerId = c.CustomerId
  JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
  JOIN Track t ON il.TrackId = t.TrackId
  JOIN Genre g ON t.GenreId = g.GenreId
  GROUP BY year_month, genre_id, country
), with_lags AS (
  SELECT *,
         LAG(year_month) OVER (PARTITION BY genre_id, country ORDER BY year_month) AS prev_year_month,
         LAG(revenue) OVER (PARTITION BY genre_id, country ORDER BY year_month) AS prev_revenue
  FROM gm
)
SELECT genre_id, genre_name, country, year_month, prev_year_month, prev_revenue, revenue,
       CASE WHEN prev_revenue IS NULL THEN NULL
            WHEN prev_revenue = 0 THEN NULL
            ELSE ROUND((revenue - prev_revenue)*1.0/prev_revenue*100,2) END AS pct_change
FROM with_lags
WHERE (prev_revenue IS NOT NULL AND ABS((revenue - prev_revenue)*1.0/prev_revenue) >= 0.5)
   OR (prev_revenue = 0 AND revenue > 0)
ORDER BY genre_id, country, year_month;