CREATE DATABASE "maxkb";

\c "maxkb";

CREATE EXTENSION "vector";


CALL paradedb.drop_bm25('search_index');


CALL paradedb.create_bm25(
  index_name => 'search_index',
  table_name => 'paragraph',
  key_field => 'id',
  text_fields => '{
    title_vector: {tokenizer: {type: "whitespace"}}, 
    content_vector: {tokenizer: {type: "whitespace"}}
  }'
);