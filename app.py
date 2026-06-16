import streamlit as st
import streamlit.components.v1 as components
import json
from google import genai
from google.genai import types
from neo4j import GraphDatabase
from pyvis.network import Network

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(page_title="Hybrid RAG Memory", layout="wide")

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
NEO4J_USER = "neo4j"

@st.cache_resource
def init_connections():
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    db_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return ai_client, db_driver

client, driver = init_connections()

# ==========================================
# 2. CÁC HÀM XỬ LÝ LÕI (HYBRID GRAPH & VECTOR)
# ==========================================

# MỚI: Hàm biến đổi text thành Vector
# MỚI: Hàm biến đổi text thành Vector (Đã sửa lỗi 404)
def get_embedding(text):
    # Cấu hình ép Gemini trả về chuẩn 768 chiều để khớp hoàn toàn với Neo4j Index
    config = types.EmbedContentConfig(output_dimensionality=768)
    
    response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=text,
        config=config
    )
    return response.embeddings[0].values

def extract_graph_data(text):
    prompt = """
    Bạn là hệ thống trích xuất đồ thị. BẮT BUỘC trả về JSON:
    {
      "nodes": [{"id": "tên", "label": "Nhãn"}],
      "edges": [{"source": "id1", "target": "id2", "type": "QUAN_HỆ"}]
    }
    """
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt + f"\n\nVăn bản: {text}",
        config=config
    )
    return json.loads(response.text)

def push_to_neo4j(graph_data):
    with driver.session() as session:
        for node in graph_data.get("nodes", []):
            node_id = str(node.get('id', '')).strip()
            
            # Chặn rác sinh ra từ LLM
            if not node_id or node_id.lower() in ['none', 'null']:
                continue
            
            # Sinh Vector cho từng Node
            vector = get_embedding(node_id)
            node_label = node.get('label', 'Unknown')
            
            # Lưu ý: Thêm nhãn ":Entity" để Vector Index của Neo4j có thể nhận diện được
            query = f"""
            MERGE (n:`{node_label}`:Entity {{id: $id}})
            SET n.embedding = $vector
            """
            session.run(query, id=node_id, vector=vector)
            
        for edge in graph_data.get("edges", []):
            source_id = str(edge.get('source', '')).strip()
            target_id = str(edge.get('target', '')).strip()
            
            if not source_id or not target_id or source_id.lower() == 'none' or target_id.lower() == 'none':
                continue
                
            query = f"""
            MATCH (source {{id: $source_id}}), (target {{id: $target_id}})
            MERGE (source)-[r:`{edge['type']}`]->(target)
            """
            session.run(query, source_id=source_id, target_id=target_id)

# MỚI: Truy xuất ngữ cảnh bằng Vector Search kết hợp Graph
def get_hybrid_context(user_question):
    # Bước 1: Biến câu hỏi của người dùng thành Vector
    question_vector = get_embedding(user_question)
    
    # Bước 2: Dùng HNSW Index tìm 3 Node có ý nghĩa sát nhất với câu hỏi, sau đó quét các láng giềng của chúng
    query = """
    CALL db.index.vector.queryNodes('entity_embedding', 3, $question_vector)
    YIELD node AS startNode, score
    MATCH (startNode)-[r]-(neighbor)
    RETURN startNode.id AS source, type(r) AS relation, neighbor.id AS target, score
    LIMIT 15
    """
    context_sentences = []
    with driver.session() as session:
        result = session.run(query, question_vector=question_vector)
        for record in result:
            context_sentences.append(f"{record['source']} -[{record['relation']}]-> {record['target']} (Độ khớp: {record['score']:.2f})")
            
    if not context_sentences:
        return "Không có thông tin liên quan trong quá khứ."
    return "\n".join(list(set(context_sentences)))

def generate_final_answer(user_input, context):
    prompt = f"""
    Dựa vào NGỮ CẢNH TỪ TRÍ NHỚ ĐỒ THỊ sau đây, hãy trả lời người dùng.
    ---
    NGỮ CẢNH: 
    {context}
    ---
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt + f"Câu hỏi của người dùng: {user_input}"
    )
    return response.text

def generate_graph_html():
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.barnes_hut(spring_length=150, damping=0.85)
    
    with driver.session() as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n.id AS s, labels(n)[0] AS sl, type(r) AS rel, m.id AS t, labels(m)[0] AS tl LIMIT 100")
        for rec in result:
            source, target, relation = str(rec["s"]), str(rec["t"]), str(rec["rel"])
            net.add_node(source, label=source, title=str(rec["sl"]), color="#00ffcc")
            net.add_node(target, label=target, title=str(rec["tl"]), color="#ff66b2")
            net.add_edge(source, target, label=relation, color="#aaaaaa")
            
    net.save_graph("temp_graph.html")
    with open("temp_graph.html", "r", encoding="utf-8") as f:
        return f.read()

# ==========================================
# 3. GIAO DIỆN WEB (UI)
# ==========================================
st.title("🧠 Hybrid GraphRAG (Vector + Graph)")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Trò chuyện với AI")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Nhập dữ liệu hoặc đặt câu hỏi...")
    
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant"):
            with st.spinner("Đang tính toán Vector và duyệt Đồ thị..."):
                # 1. Bóc tách và đẩy lên Database kèm Vector
                extracted_data = extract_graph_data(user_input)
                push_to_neo4j(extracted_data)
                
                # 2. Truy xuất bằng Hybrid Search
                context = get_hybrid_context(user_input)
                
                # 3. Trả lời
                answer = generate_final_answer(user_input, context)
                st.markdown(answer)
                
                with st.expander("🔍 Xem thông số Hybrid RAG"):
                    st.text("Dữ liệu bóc tách được (Mới):")
                    st.json(extracted_data)
                    st.text("Ngữ cảnh kéo về (Dựa trên Vector Similarity):")
                    st.text(context)
                    
        st.session_state.messages.append({"role": "assistant", "content": answer})

with col2:
    st.subheader("Mạng lưới Tri thức")
    components.html(generate_graph_html(), height=620)