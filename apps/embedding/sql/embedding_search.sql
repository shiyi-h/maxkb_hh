SELECT
    paragraph_id,
	comprehensive_score,
	comprehensive_score as similarity
FROM
	(
	SELECT paragraph_id, max(similarity) as comprehensive_score
	FROM
		( SELECT paragraph_id, ( 1 - ( embedding.embedding <=>  %s ) ) AS similarity FROM embedding ${embedding_query}) TEMP
	GROUP BY 
		paragraph_id
	) DISTINCT_TEMP
WHERE comprehensive_score>%s
ORDER BY comprehensive_score DESC
LIMIT %s