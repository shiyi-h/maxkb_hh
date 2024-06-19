SELECT
	paragraph_id,
	comprehensive_score,
	comprehensive_score AS similarity
FROM
	(
	SELECT 
		paragraph_id,
		max(similarity) AS comprehensive_score
	FROM
		(
		SELECT
			paragraph_id,
			(( 1 - ( embedding.embedding <=>  %s ) )*0.7+0.3*ts_rank_cd( embedding.search_vector, to_tsquery('simple', %s ), 32 )) AS similarity
		FROM
			embedding ${embedding_query}
		) TEMP
	GROUP BY
		paragraph_id
	) DISTINCT_TEMP
WHERE
	comprehensive_score >%s
ORDER BY
	comprehensive_score DESC
	LIMIT %s