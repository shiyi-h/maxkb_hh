with vec as (
SELECT
   paragraph_id,score
FROM
	(
	SELECT paragraph_id, max(similarity) as score
	FROM
		( SELECT paragraph_id, ( 1 - ( embedding.embedding <=>  %s) ) AS similarity FROM embedding ${embedding_query}) TEMP
	GROUP BY 
		paragraph_id
	) DISTINCT_TEMP
WHERE score>0.1
),
bm25 as (
select paragraph_id, score/sqrt(sum(p_score) over()) as score
from 
	(
	select paragraph_id, score, power(score,2) as p_score
	from 
		(
			SELECT id as paragraph_id, max(paradedb.rank_bm25(id)) as score
			FROM search_index.search(%s) ${embedding_query}
			group by id 
		) t0 
	) t1 
)
select paragraph_id,
comprehensive_score,
comprehensive_score as similarity 
from 
(select 
coalesce(vec.paragraph_id, bm25.paragraph_id) as paragraph_id,
COALESCE(vec.score, 0.0) + COALESCE(bm25.score, 0.0) * 0.5 as comprehensive_score
FROM vec
FULL OUTER JOIN bm25 
ON vec.paragraph_id = bm25.paragraph_id) t 
where comprehensive_score>%s 
order by comprehensive_score desc 
limit %s
