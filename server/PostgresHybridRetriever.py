import psycopg2
import psycopg2.extras
from typing import List
import regex
import nltk
import os

class PostgresHybridRetriever():
    def __init__(self, connection_pool):
        self.connection_pool = connection_pool
    
    def setup_database(self, embedding_dimension):
        conn = None
        conn = self.connection_pool.getconn()
        with conn.cursor() as cursor:
            # Setup database if need be
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS ragmeup_dense_embeddings (
                    id VARCHAR(32) PRIMARY KEY,
                    embedding public.vector({embedding_dimension}) NOT NULL,
                    content varchar NOT NULL,
                    metadata jsonb NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ragmeup_dense_embedding_index ON ragmeup_dense_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64');""")
            cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS ragmeup_sparse_embeddings (
                        id VARCHAR(32) PRIMARY KEY,
                        content TEXT,
                        metadata JSONB
                    );""")
            # Create BM25 index
            cursor.execute(fr"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE c.relname = lower('ragmeup_sparse_embeddings_bm25')
                        AND n.nspname = 'public'
                        AND c.relkind = 'i'
                    ) THEN
                        CREATE INDEX ragmeup_sparse_embeddings_bm25 ON ragmeup_sparse_embeddings USING bm25 (id, content) WITH (key_field='id');
                        CREATE INDEX idx_metadata_dense_dataset ON ragmeup_dense_embeddings (((metadata::jsonb ->> 'dataset')::text));
                        CREATE INDEX idx_metadata_sparse_dataset ON ragmeup_sparse_embeddings (((metadata::jsonb->>'dataset')::text));
                    END IF;
                END $$;
            """)
            conn.commit()
        
        self.connection_pool.putconn(conn)

    def add_documents(self, documents) -> List[str]:
        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor() as cursor:
                # Add to the BM25 index
                records = [
                    (doc['id'], doc['content'], doc['metadata'])
                    for doc in documents
                ]
                psycopg2.extras.execute_batch(
                    cursor,
                    f"""
                        INSERT INTO ragmeup_sparse_embeddings (id, content, metadata)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """,
                    records
                )
                conn.commit()
            
                # Now the dense embeddings
                records = [
                    (doc['id'], doc['embedding'].tolist(), doc['content'], doc['metadata'])
                    for doc in documents
                ]
                psycopg2.extras.execute_batch(
                    cursor,
                    f"""
                        INSERT INTO ragmeup_dense_embeddings (id, embedding, content, metadata)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """,
                    records
                )
                conn.commit()
        except Exception as e:
            print(f"Error executing Postgres query while inserting documents: {e}")
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def has_data(self):
        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM ragmeup_sparse_embeddings;")
                return cursor.fetchone()[0] > 0
        except Exception as e:
            print(f"Error while checking if Postgres has data: {e}")
            return False
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def escape_query(self, query):
        # Tokenize the query into words
        tokens = nltk.word_tokenize(query)
        # Use regex to keep alphanumeric characters and diacritics, remove all others
        tokens = [regex.sub(r'[^\p{L}\p{N}\s]', '', token) for token in tokens]
        # Rejoin tokens into a sanitized string
        sanitized_query = ' '.join(tokens)
        return sanitized_query

    def get_relevant_documents(self, query, query_embedding):
        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor() as cursor:
                # Remove the re2 prompt if it exists
                if os.getenv("use_re2") == "True":
                    os.getenv("re2_prompt")
                    index = query.find(f"\n{os.getenv('re2_prompt')}")
                    query = query[:index]
                
                # Get both the dense and sparse results, unified
                search_command = f"""
                    WITH combined AS (
                        SELECT * FROM (
                            SELECT
                                id,
                                content,
                                metadata,
                                paradedb.score(id) AS score_bm25,
                                NULL::float AS distance,
                                'bm25' AS source
                            FROM ragmeup_sparse_embeddings
                            WHERE content @@@ %s
                            ORDER BY score_bm25 DESC
                            LIMIT %s
                        ) bm25_results

                        UNION ALL

                        SELECT * FROM (
                            SELECT
                                id,
                                content,
                                metadata,
                                NULL::float AS score_bm25,
                                embedding <=> %s::vector AS distance,
                                'vector' AS source
                            FROM ragmeup_dense_embeddings
                            ORDER BY distance
                            LIMIT %s
                        ) vector_results
                    ),
                    deduplicated AS (
                        SELECT 
                            id,
                            content,
                            metadata,
                            MAX(score_bm25) as score_bm25,
                            MIN(distance) as distance,
                            string_agg(source, ',') as sources
                        FROM combined
                        GROUP BY id, content, metadata
                    ),
                    scored AS (
                        SELECT *,
                            MAX(score_bm25) OVER () AS max_bm25,
                            MIN(distance) OVER () AS min_distance,
                            MAX(distance) OVER () AS max_distance
                        FROM deduplicated
                    )
                    SELECT
                        id, content, metadata, score_bm25, distance, sources,
                        (
                            0.5 * COALESCE(score_bm25 / NULLIF(max_bm25, 0), 0) +
                            0.5 * COALESCE(1 - (distance - min_distance) / NULLIF((max_distance - min_distance), 0), 0)
                        ) AS hybrid_score
                    FROM scored
                    ORDER BY hybrid_score DESC
                    LIMIT %s;
                """
                cursor.execute(search_command, (
                    self.escape_query(query),
                    int(os.getenv("vector_store_k")),
                    query_embedding.tolist(),
                    int(os.getenv("vector_store_k")),
                    int(os.getenv("vector_store_k")),
                ))
                
                results = cursor.fetchall()

                results = [{
                    "content": row[1],
                    "metadata": {**row[2], "distance": float(row[6])}
                } for row in results]
                return results
        except Exception as e:
            print(f"Error while getting relevant documents from Postgres: {e}")
        finally:
            # Return the connection to the pool
            if conn:
                self.connection_pool.putconn(conn)

    def delete(self, filenames: List[str]) -> None:
        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor() as cursor:
                placeholders = ','.join(['%s'] * len(filenames))
                cursor.execute(f"DELETE FROM ragmeup_sparse_embeddings WHERE metadata->>'source' IN ({placeholders});", tuple(filenames))
                conn.commit()
                delete_count = cursor.rowcount
                cursor.execute(f"DELETE FROM ragmeup_dense_embeddings WHERE metadata->>'source' IN ({placeholders});", tuple(filenames))
                conn.commit()
                
                # Close the current transaction before running VACUUM
                conn.close()
                return delete_count
        except Exception as e:
            print(f"Error while deleting documents from Postgres: {e}")
            if conn:
                conn.rollback()
        finally:
            # Return the connection to the pool
            if conn:
                self.connection_pool.putconn(conn)
    
    def get_all_document_names(self):
        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("select distinct metadata->>'source', metadata->>'dataset' from ragmeup_sparse_embeddings;")
                results = cursor.fetchall()
                return [{"filename": row[0].replace(f'{os.getenv("data_directory")}/', ""), "dataset": row[1]} for row in results]
                conn.commit()
        except Exception as e:
            print(f"Error while getting all document names from Postgres: {e}")
            if conn:
                conn.rollback()
        finally:
            # Return the connection to the pool
            if conn:
                self.connection_pool.putconn(conn)

    def close(self):
        try:
            self.connection_pool.closeall()
            print("Connection pool closed.")
        except Exception as e:
            print(f"Error closing connection pool: {e}")