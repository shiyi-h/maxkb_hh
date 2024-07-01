with bm25 as (
select paragraph_id, score/sqrt(sum(p_score) over()) as score
from 
	(
	select paragraph_id, score, power(score,2) as p_score
	from 
		(
			SELECT id as paragraph_id, max(paradedb.rank_bm25(id)) as score
			FROM search_index.search(%s) ${keywords_query}
			group by id 
		) t0 
	) t1 
)
select paragraph_id,
score as comprehensive_score, 
score as similarity
from 
bm25 
where score>%s 
order by score desc 
limit %s