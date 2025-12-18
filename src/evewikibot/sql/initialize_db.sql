CREATE EXTENSION vector;

CREATE TABLE eve_wiki (id bigserial PRIMARY KEY, embedding vector(768),chunk text, title text, page_url text);