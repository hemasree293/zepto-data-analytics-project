-- Q1_SELECT_WHERE
SELECT title, price_gbp, rating FROM books WHERE rating >= 4 ORDER BY rating DESC, price_gbp DESC;

-- Q2_ORDER_BY_LIMIT
SELECT title, price_gbp, price_inr FROM books ORDER BY price_gbp DESC LIMIT 10;

-- Q3_DISTINCT
SELECT DISTINCT category_name FROM categories ORDER BY category_name;

-- Q4_IN
SELECT title, rating, category_id FROM books WHERE rating IN (4, 5) ORDER BY rating DESC, title LIMIT 15;

-- Q5_BETWEEN
SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 20 AND 40 ORDER BY price_gbp;

-- Q6_JOIN_TOP_RATED_PER_CATEGORY
WITH ranked AS (
SELECT b.title,b.price_gbp,b.rating,b.in_stock,c.category_name,
ROW_NUMBER() OVER (PARTITION BY c.category_id ORDER BY b.rating DESC,b.price_gbp DESC,b.title) AS rn
FROM books b JOIN categories c ON b.category_id=c.category_id)
SELECT category_name,title,price_gbp,rating,in_stock FROM ranked WHERE rn<=10
ORDER BY category_name,rating DESC,price_gbp DESC,title;